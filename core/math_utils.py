import numpy as np


def solve_dare(
    A,
    B,
    Q,
    R,
    max_iter=2000,
    tol=1e-10
):
    """
    Iterative solver for the discrete algebraic Riccati equation.

    Solves:

    P = Q + AᵀPA - AᵀPB(R + BᵀPB)⁻¹BᵀPA
    """

    # ======================================================
    # INPUT VALIDATION
    # ======================================================

    if A.ndim != 2 or A.shape[0] != A.shape[1]:
        raise ValueError("A must be square")

    n = A.shape[0]

    if Q.shape != (n, n):
        raise ValueError("Q has invalid dimensions")

    if B.shape[0] != n:
        raise ValueError("B has incompatible dimensions")

    m = B.shape[1]

    if R.shape != (m, m):
        raise ValueError("R has invalid dimensions")

    # ======================================================
    # INITIALIZATION
    # ======================================================

    P = Q.copy()

    info = {
        "converged": False,
        "iterations": 0,
        "error": np.inf
    }

    # ======================================================
    # ITERATIVE RICCATI SOLVER
    # ======================================================

    for i in range(max_iter):

        S = R + B.T @ P @ B

        # singularity check
        if np.linalg.det(S) == 0:
            raise np.linalg.LinAlgError(
                "R + BᵀPB is singular"
            )

        K = np.linalg.inv(S)

        P_new = (
            Q
            + A.T @ P @ A
            - A.T @ P @ B @ K @ B.T @ P @ A
        )

        err = np.max(np.abs(P_new - P))

        if err < tol:

            info["converged"] = True
            info["iterations"] = i + 1
            info["error"] = err

            return P_new, info

        P = P_new

    # ======================================================
    # NO CONVERGENCE
    # ======================================================

    info["iterations"] = max_iter
    info["error"] = err

    return P, info