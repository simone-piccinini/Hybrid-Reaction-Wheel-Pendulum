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


# --------------------------------------------------------------------------- #
# eigenvalues — general real matrices via the shifted-QR algorithm
# --------------------------------------------------------------------------- #
# Approved as the general (complex-capable) extension of numerical_standards.md
# §3's eigen-decomposition row (architecture decision, 2026-05-30): needed for
# model.md's open-loop instability check and lqr.md's closed-loop unit-circle
# test. LAPACK's xHSEQR budgets 30 QR sweeps per eigenvalue; matrices here are
# small, so we are generous.
EIG_MAX_SWEEPS_PER_EIGENVALUE: int = 100

# After this many sweeps without a deflation, take one ad-hoc ("exceptional")
# shift to break the cycles on which the Wilkinson shift stalls (e.g.
# permutation matrices) — the standard LAPACK safeguard.
_EIG_EXCEPTIONAL_EVERY: int = 10

_MACHINE_EPS: float = float(np.finfo(np.float64).eps)


class EigNotConverged(Exception):
    """Raised when the shifted-QR iteration exhausts its sweep budget."""


def _hessenberg(A: np.ndarray) -> np.ndarray:
    """Householder reduction to upper Hessenberg form (GVL Alg. 7.4.2).

    Returns ``H = Qᵀ A Q`` with ``H[i, j] = 0`` for ``i > j + 1`` and ``Q``
    orthogonal, so ``H`` has the same eigenvalues as ``A``.
    """
    H = A.copy()
    n = H.shape[0]
    for k in range(n - 2):
        x = H[k + 1 :, k]
        sigma = math.sqrt(float(x @ x))
        if sigma == 0.0:
            continue  # column already reduced
        # v = x − α e₁ with α = −sign(x₀)·‖x‖ avoids cancellation (GVL §5.1.3)
        alpha = -sigma if x[0] >= 0.0 else sigma
        v = x.copy()
        v[0] -= alpha
        beta = float(v @ v)
        if beta == 0.0:
            continue
        # similarity transform with P = I − (2/β) v vᵀ: H ← P H P
        w = (2.0 / beta) * (v @ H[k + 1 :, k:])
        H[k + 1 :, k:] -= np.outer(v, w)
        u = (2.0 / beta) * (H[:, k + 1 :] @ v)
        H[:, k + 1 :] -= np.outer(u, v)
        H[k + 2 :, k] = 0.0  # exact zeros below the subdiagonal
    return H


def _eig2x2(a: complex, b: complex, c: complex, d: complex) -> tuple[complex, complex]:
    """Both eigenvalues of ``[[a, b], [c, d]]``: ``(a+d)/2 ± √(((a−d)/2)² + bc)``."""
    mid = 0.5 * (a + d)
    disc = np.sqrt(np.complex128((0.5 * (a - d)) ** 2 + b * c))
    return mid + disc, mid - disc


def _wilkinson_shift(a: complex, b: complex, c: complex, d: complex) -> complex:
    """The eigenvalue of the trailing 2×2 block closer to ``d`` (GVL §7.5.1)."""
    mu1, mu2 = _eig2x2(a, b, c, d)
    return mu1 if abs(mu1 - d) <= abs(mu2 - d) else mu2


def _givens(f: complex, g: complex) -> tuple[float, complex]:
    """Unitary ``G = [[c, s], [−s̄, c]]`` (``c`` real ≥ 0) with ``G[f, g]ᵀ = [r, 0]ᵀ``.

    The complex Givens rotation in LAPACK's ``clartg`` convention (GVL §5.1.8).
    """
    if g == 0.0:
        return 1.0, 0.0 + 0.0j
    if f == 0.0:
        return 0.0, complex(np.conj(g) / abs(g))
    denom = math.hypot(abs(f), abs(g))
    c = abs(f) / denom
    s = (f / abs(f)) * np.conj(g) / denom
    return c, complex(s)


def _qr_sweep(H: np.ndarray, lo: int, hi: int, mu: complex) -> None:
    """One shifted QR step ``H ← R Q + μI`` on the active block ``lo..hi``.

    ``Q R = H − μI`` is formed implicitly with Givens rotations on the
    Hessenberg block (GVL §7.5, "Hessenberg QR step"); the similarity
    transform preserves eigenvalues and the Hessenberg structure.
    """
    T = H[lo : hi + 1, lo : hi + 1].copy()
    m = T.shape[0]
    T[np.diag_indices(m)] -= mu
    rotations: list[tuple[float, complex]] = []
    for k in range(m - 1):
        c, s = _givens(T[k, k], T[k + 1, k])
        rotations.append((c, s))
        row_k = c * T[k, :] + s * T[k + 1, :]
        row_k1 = -np.conj(s) * T[k, :] + c * T[k + 1, :]
        T[k, :], T[k + 1, :] = row_k, row_k1
    for k, (c, s) in enumerate(rotations):  # T ← T G₀ᴴ G₁ᴴ … = R Q
        col_k = c * T[:, k] + np.conj(s) * T[:, k + 1]
        col_k1 = -s * T[:, k] + c * T[:, k + 1]
        T[:, k], T[:, k + 1] = col_k, col_k1
    T[np.diag_indices(m)] += mu
    H[lo : hi + 1, lo : hi + 1] = T


def eigvals(M, max_sweeps: int | None = None) -> np.ndarray:
    """All eigenvalues of a real square matrix, in no particular order.

    The practical QR algorithm (GVL §7.5): Householder reduction to upper
    Hessenberg form, then Wilkinson-shifted QR iterations in complex
    arithmetic with deflation — complex conjugate pairs emerge through the
    complex shift, avoiding real-Schur 2×2 bookkeeping. Trailing 1×1 and 2×2
    blocks deflate directly. For defective (repeated, non-diagonalisable)
    eigenvalues the attainable accuracy degrades to ~√ε — inherent to the
    problem, not the algorithm.

    Parameters
    ----------
    M : (n, n) array_like
        Real square matrix with finite entries.
    max_sweeps : int, optional
        Total QR-sweep budget (default ``EIG_MAX_SWEEPS_PER_EIGENVALUE · n``).

    Returns
    -------
    (n,) ndarray of complex128
        The eigenvalues (complex even when all are real).

    Raises
    ------
    ValueError
        If ``M`` is not square or contains non-finite entries.
    EigNotConverged
        If the sweep budget is exhausted before full deflation.
    """
    A = _as_square(M)
    if not np.all(np.isfinite(A)):
        raise ValueError("eigvals requires finite entries")
    n = A.shape[0]
    if n == 0:
        return np.empty(0, dtype=np.complex128)
    budget = EIG_MAX_SWEEPS_PER_EIGENVALUE * n if max_sweeps is None else max_sweeps
    H = _hessenberg(A).astype(np.complex128)
    scale = float(np.max(np.abs(H))) or 1.0  # negligibility floor for zero rows
    eig = np.empty(n, dtype=np.complex128)
    hi = n - 1
    sweeps_total = 0
    sweeps_since_deflation = 0

    def negligible(i: int) -> bool:
        # standard relative criterion: |h_{i,i-1}| ≤ ε(|h_{i-1,i-1}| + |h_{ii}|)
        tol = _MACHINE_EPS * (abs(H[i - 1, i - 1]) + abs(H[i, i]))
        return abs(H[i, i - 1]) <= (tol if tol > 0.0 else _MACHINE_EPS * scale)

    while hi >= 0:
        if hi == 0:
            eig[0] = H[0, 0]
            break
        if negligible(hi):  # 1×1 deflation at the bottom
            H[hi, hi - 1] = 0.0
            eig[hi] = H[hi, hi]
            hi -= 1
            sweeps_since_deflation = 0
            continue
        lo = hi
        while lo > 0 and not negligible(lo):
            lo -= 1
        if lo > 0:
            H[lo, lo - 1] = 0.0
        if hi - lo == 1:  # 2×2 deflation in closed form
            eig[lo], eig[hi] = _eig2x2(H[lo, lo], H[lo, hi], H[hi, lo], H[hi, hi])
            hi = lo - 1
            sweeps_since_deflation = 0
            continue
        if sweeps_total >= budget:
            raise EigNotConverged(
                f"shifted QR did not deflate within {budget} sweeps"
            )
        sweeps_total += 1
        sweeps_since_deflation += 1
        if sweeps_since_deflation % _EIG_EXCEPTIONAL_EVERY == 0:
            mu = H[hi, hi] + abs(H[hi, hi - 1])  # exceptional shift
        else:
            mu = _wilkinson_shift(
                H[hi - 1, hi - 1], H[hi - 1, hi], H[hi, hi - 1], H[hi, hi]
            )
        _qr_sweep(H, lo, hi, mu)
    return eig


def spectral_radius(M, max_sweeps: int | None = None) -> float:
    """``ρ(M) = max |λᵢ|`` over the eigenvalues of ``M``.

    The quantity in lqr.md's closed-loop caution (all eigenvalues of
    ``A_d − B_d K`` strictly inside the unit circle ⇔ ``ρ < 1``) and model.md's
    open-loop instability check.
    """
    e = eigvals(M, max_sweeps=max_sweeps)
    return float(np.max(np.abs(e))) if e.size else 0.0
