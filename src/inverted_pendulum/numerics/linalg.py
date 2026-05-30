"""Hand-written dense linear-algebra primitives.

Pure NumPy: array storage, broadcasting, ``.T``, slicing, and the matmul
operator ``@`` only. No call to ``numpy.linalg`` (or any other banned solver)
appears in this module — see ``AGENTS.md`` §3 and
``docs/conventions/numerical_standards.md`` §3. Each routine is implemented
from the underlying algorithm and cross-checked against a reference library in
``tests/validation/`` to ``VALIDATION_RTOL``.

References
----------
Golub & Van Loan, *Matrix Computations*, 4th ed. (cited as "GVL Alg. x.y.z").
docs/conventions/numerical_standards.md §§3-4.
"""

from __future__ import annotations

import math
from typing import NamedTuple

import numpy as np

from .constants import ATOL, SYM_TOL


class NotPositiveDefiniteError(Exception):
    """Raised when a Cholesky pivot is non-positive (matrix not positive-definite)."""


class SingularMatrixError(Exception):
    """Raised when an LU or triangular system has a (near-)zero pivot."""


class LUFactors(NamedTuple):
    """Partial-pivot LU factorisation with row permutation.

    Invariant: ``M[perm] == L @ U`` (equivalently ``P M = L U`` with ``P`` the
    row permutation encoded by ``perm``), ``L`` unit lower-triangular and ``U``
    upper-triangular.
    """

    L: np.ndarray
    U: np.ndarray
    perm: np.ndarray


def _as_square(M) -> np.ndarray:
    """Coerce to a square float64 2-D array; raise on the wrong shape (§1)."""
    A = np.asarray(M, dtype=np.float64)
    if A.ndim != 2 or A.shape[0] != A.shape[1]:
        raise ValueError(f"expected a square 2-D matrix, got shape {A.shape}")
    return A


def max_abs_asymmetry(M) -> float:
    """Return ``‖M − Mᵀ‖∞``, the largest absolute asymmetry.

    numerical_standards.md §4 (symmetry diagnostic).
    """
    A = _as_square(M)
    if A.size == 0:
        return 0.0
    return float(np.max(np.abs(A - A.T)))


def is_symmetric(M, tol: float = SYM_TOL) -> bool:
    """True iff ``‖M − Mᵀ‖∞ ≤ tol`` (default ``SYM_TOL``); numerical_standards.md §4."""
    return max_abs_asymmetry(M) <= tol


def symmetrize(M) -> np.ndarray:
    """Return the symmetric part ``½(M + Mᵀ)``.

    numerical_standards.md §4: symmetrise a matrix expected to be symmetric
    before factoring it, to remove round-off asymmetry.
    """
    A = _as_square(M)
    return 0.5 * (A + A.T)


def cholesky(M, jitter: float = 0.0) -> np.ndarray:
    """Cholesky factorisation ``M = L Lᵀ`` with lower-triangular ``L``.

    Column (Cholesky–Banachiewicz) algorithm, GVL Alg. 4.2.1;
    numerical_standards.md §3. An optional non-negative ``jitter`` is added to
    the diagonal before factoring (the SPD-conditioning guard of
    numerical_standards.md §4); any geometric escalation of that jitter is the
    caller's responsibility.

    Parameters
    ----------
    M : (n, n) array_like
        Symmetric positive-definite matrix (symmetric within ``SYM_TOL``).
    jitter : float
        Non-negative value added to the diagonal prior to factorisation.

    Returns
    -------
    (n, n) ndarray
        Lower-triangular ``L`` with positive diagonal such that ``L @ L.T == M``
        (up to round-off, plus ``jitter`` on the diagonal if supplied).

    Raises
    ------
    NotPositiveDefiniteError
        If a non-positive pivot is encountered (``M`` not positive-definite).
    ValueError
        If ``M`` is not square, not symmetric within ``SYM_TOL``, or
        ``jitter`` is negative.
    """
    A = _as_square(M)
    if not is_symmetric(A):
        raise ValueError(
            "cholesky requires a symmetric matrix (within SYM_TOL); "
            "call symmetrize() first"
        )
    if jitter < 0.0:
        raise ValueError("jitter must be non-negative")
    n = A.shape[0]
    if jitter:
        A = A + jitter * np.eye(n)
    L = np.zeros((n, n), dtype=np.float64)
    for j in range(n):
        # Diagonal: L[j,j] = sqrt(A[j,j] - sum_{k<j} L[j,k]^2).
        s = A[j, j] - L[j, :j] @ L[j, :j]
        if s <= 0.0:
            raise NotPositiveDefiniteError(
                f"non-positive pivot {s!r} at index {j}; matrix is not positive-definite"
            )
        L[j, j] = math.sqrt(s)
        # Below-diagonal column j (vectorised over rows i > j).
        if j + 1 < n:
            L[j + 1 :, j] = (A[j + 1 :, j] - L[j + 1 :, :j] @ L[j, :j]) / L[j, j]
    return L


def solve_lower(L, b) -> np.ndarray:
    """Solve ``L x = b`` by forward substitution (``L`` lower-triangular).

    GVL Alg. 3.1.1; numerical_standards.md §3. ``b`` may be a vector ``(n,)`` or
    a matrix ``(n, m)`` (its columns are solved simultaneously).

    Raises
    ------
    SingularMatrixError
        If a diagonal entry of ``L`` is zero within ``ATOL``.
    """
    Lm = _as_square(L)
    rhs = np.asarray(b, dtype=np.float64)
    n = Lm.shape[0]
    if rhs.shape[0] != n:
        raise ValueError(f"shape mismatch: L is {Lm.shape}, b is {rhs.shape}")
    x = np.zeros(rhs.shape, dtype=np.float64)
    for i in range(n):
        if abs(Lm[i, i]) <= ATOL:
            raise SingularMatrixError(f"zero pivot at index {i} in lower solve")
        x[i] = (rhs[i] - Lm[i, :i] @ x[:i]) / Lm[i, i]
    return x


def solve_upper(U, b) -> np.ndarray:
    """Solve ``U x = b`` by back substitution (``U`` upper-triangular).

    GVL Alg. 3.1.2; numerical_standards.md §3. ``b`` may be ``(n,)`` or ``(n, m)``.

    Raises
    ------
    SingularMatrixError
        If a diagonal entry of ``U`` is zero within ``ATOL``.
    """
    Um = _as_square(U)
    rhs = np.asarray(b, dtype=np.float64)
    n = Um.shape[0]
    if rhs.shape[0] != n:
        raise ValueError(f"shape mismatch: U is {Um.shape}, b is {rhs.shape}")
    x = np.zeros(rhs.shape, dtype=np.float64)
    for i in range(n - 1, -1, -1):
        if abs(Um[i, i]) <= ATOL:
            raise SingularMatrixError(f"zero pivot at index {i} in upper solve")
        x[i] = (rhs[i] - Um[i, i + 1 :] @ x[i + 1 :]) / Um[i, i]
    return x


def chol_solve(M, b, jitter: float = 0.0) -> np.ndarray:
    """Solve the SPD system ``M x = b`` via Cholesky: ``x = Lᵀ \\ (L \\ b)``.

    numerical_standards.md §3 (the ``chol_solve`` primitive consumed by the GP
    and Kalman layers). Factor once with :func:`cholesky`, then one forward and
    one back substitution. ``b`` may be ``(n,)`` or ``(n, m)``.
    """
    L = cholesky(M, jitter=jitter)
    y = solve_lower(L, b)
    return solve_upper(L.T, y)


def lu_factor(M) -> LUFactors:
    """Partial-pivot LU factorisation ``P M = L U``.

    Outer-product Gaussian elimination with partial (row) pivoting,
    GVL Alg. 3.4.1; numerical_standards.md §3. Returns unit-lower ``L``, upper
    ``U`` and the row permutation ``perm`` with ``M[perm] == L @ U``.

    Raises
    ------
    SingularMatrixError
        If the largest available pivot in some column is zero within ``ATOL``.
    """
    U = _as_square(M).copy()
    n = U.shape[0]
    L = np.eye(n, dtype=np.float64)
    perm = np.arange(n)
    for k in range(n):
        # Partial pivot: largest-magnitude entry in column k at or below row k.
        p = k + int(np.argmax(np.abs(U[k:, k])))
        if abs(U[p, k]) <= ATOL:
            raise SingularMatrixError(f"singular matrix: zero pivot in column {k}")
        if p != k:
            U[[k, p], :] = U[[p, k], :]
            L[[k, p], :k] = L[[p, k], :k]  # carry along multipliers already stored
            perm[[k, p]] = perm[[p, k]]
        # Eliminate below the pivot; store multipliers in L.
        L[k + 1 :, k] = U[k + 1 :, k] / U[k, k]
        U[k + 1 :, k:] -= np.outer(L[k + 1 :, k], U[k, k:])
    return LUFactors(L=L, U=U, perm=perm)


def lu_solve(factors: LUFactors, b) -> np.ndarray:
    """Solve ``M x = b`` from an :class:`LUFactors` produced by :func:`lu_factor`.

    GVL §3.4: permute (``P b``), forward-substitute ``L y = P b`` then
    back-substitute ``U x = y``. ``b`` may be ``(n,)`` or ``(n, m)``.
    """
    rhs = np.asarray(b, dtype=np.float64)
    if rhs.shape[0] != factors.U.shape[0]:
        raise ValueError(
            f"shape mismatch: factors are {factors.U.shape}, b is {rhs.shape}"
        )
    pb = rhs[factors.perm]
    y = solve_lower(factors.L, pb)
    return solve_upper(factors.U, y)


def lu_solve_matrix(M, b) -> np.ndarray:
    """Convenience: factor ``M`` (partial-pivot LU) and solve ``M x = b``.

    Equivalent to ``lu_solve(lu_factor(M), b)``; numerical_standards.md §3
    "general linear solve (LU)".
    """
    return lu_solve(lu_factor(M), b)
