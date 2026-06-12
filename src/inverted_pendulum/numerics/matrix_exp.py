"""Matrix exponential via scaling-and-squaring with a Padé approximant.

Pure NumPy (``@``, broadcasting, reductions); the implicit matrix inverse in
the Padé approximant is applied through the hand-written LU solve in
:mod:`inverted_pendulum.numerics.linalg`, never an explicit inverse and never a
``numpy.linalg`` call (``AGENTS.md`` §3, ``numerical_standards.md`` §3).

Reference
---------
N. J. Higham, "The Scaling and Squaring Method for the Matrix Exponential
Revisited", SIAM J. Matrix Anal. Appl. 26(4):1179-1193 (2005) — the degree-13
diagonal Padé approximant (§§2-3) with ℓ1-norm-based scaling.
"""

from __future__ import annotations

import math

import numpy as np

from .linalg import lu_solve_matrix

# Diagonal [13/13] Padé numerator coefficients b_0 … b_13 of exp (Higham 2005).
_PADE13_B: tuple[float, ...] = (
    64764752532480000.0,
    32382376266240000.0,
    7771770303897600.0,
    1187353796428800.0,
    129060195264000.0,
    10559470521600.0,
    670442572800.0,
    33522128640.0,
    1323241920.0,
    40840800.0,
    960960.0,
    16380.0,
    182.0,
    1.0,
)

# ℓ1-norm threshold below which the degree-13 approximant is accurate to
# machine precision without further scaling (Higham 2005, θ_13).
_THETA_13: float = 5.371920351148152


def expm(M) -> np.ndarray:
    """Matrix exponential ``exp(M)`` by scaling-and-squaring + [13/13] Padé.

    Higham (2005), §§2-3. Pick the smallest scaling exponent ``s`` so that
    ``‖M / 2ˢ‖₁ ≤ θ₁₃``, form the degree-13 diagonal Padé approximant
    ``r₁₃(M/2ˢ) = q(·)⁻¹ p(·)`` of the scaled matrix, then square the result
    ``s`` times: ``exp(M) = r₁₃(M/2ˢ)^{2ˢ}``.

    The approximant is evaluated through Higham's ``U``/``V`` split, with
    ``p = U + V`` and ``q = V − U`` (the diagonal-Padé identity ``q(M) = p(−M)``).
    The inverse ``q⁻¹`` is applied as a general LU solve, never formed
    explicitly (``numerical_standards.md`` §3). Consumed by the dynamics layer
    to discretise ``(A, B)`` via the augmented exponential (``model.md``,
    "Discretise").

    Parameters
    ----------
    M : (n, n) array_like
        Square, finite matrix.

    Returns
    -------
    (n, n) ndarray
        ``exp(M)`` as float64.

    Raises
    ------
    ValueError
        If ``M`` is not square 2-D, or contains non-finite entries.
    """
    A = np.asarray(M, dtype=np.float64)
    if A.ndim != 2 or A.shape[0] != A.shape[1]:
        raise ValueError(f"expm expects a square 2-D matrix, got shape {A.shape}")
    if not np.all(np.isfinite(A)):
        raise ValueError("expm requires a finite matrix")
    n = A.shape[0]
    ident = np.eye(n, dtype=np.float64)

    # Scaling: bring the ℓ1-norm within the degree-13 region of accuracy.
    norm1 = float(np.max(np.sum(np.abs(A), axis=0))) if n else 0.0
    s = max(0, int(math.ceil(math.log2(norm1 / _THETA_13)))) if norm1 > _THETA_13 else 0
    A = A / (2.0**s)

    # Degree-13 Padé numerator p = U + V and denominator q = V - U (Higham 2005).
    b = _PADE13_B
    A2 = A @ A
    A4 = A2 @ A2
    A6 = A4 @ A2
    U = A @ (
        A6 @ (b[13] * A6 + b[11] * A4 + b[9] * A2)
        + b[7] * A6
        + b[5] * A4
        + b[3] * A2
        + b[1] * ident
    )
    V = (
        A6 @ (b[12] * A6 + b[10] * A4 + b[8] * A2)
        + b[6] * A6
        + b[4] * A4
        + b[2] * A2
        + b[0] * ident
    )
    P = U + V  # p_13(A)
    Q = V - U  # q_13(A) = p_13(-A)

    # exp(A_scaled) ≈ q⁻¹ p, applied as a linear solve (no explicit inverse).
    R = lu_solve_matrix(Q, P)

    # Squaring phase undoes the scaling: R ← R² repeated s times.
    for _ in range(s):
        R = R @ R
    return R
