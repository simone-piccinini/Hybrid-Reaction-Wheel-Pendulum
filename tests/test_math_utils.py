"""
test_math_utils.py
------------------

pytest suite for core.math_utils.solve_dare

Run with:
    pytest tests/test_math_utils.py -v

This suite verifies:

1. Numerical convergence
2. Symmetry of Riccati solution
3. Positive definiteness
4. Bellman equation residual
5. Closed-loop stability
6. Numerical robustness
7. Edge-case handling
8. Output contract consistency

Mathematical assumptions:
- (A,B) must be stabilizable
- Q must be positive semidefinite
- R must be positive definite
"""

import numpy as np
import pytest

from core.math_utils import solve_dare


class TestSolveDare:
    """
    Unit tests for the standalone iterative DARE solver.
    """

    # ======================================================
    # SIMPLE STABILIZABLE TEST SYSTEM
    # ======================================================

    A2 = np.array([
        [1.1, 0.0],
        [0.0, 0.9]
    ])

    B2 = np.array([
        [1.0],
        [1.0]
    ])

    Q2 = np.eye(2)

    R2 = np.array([
        [1.0]
    ])

    # ======================================================
    # OUTPUT CONTRACT
    # ======================================================

    def test_returns_P_and_info(self):

        P, info = solve_dare(
            self.A2,
            self.B2,
            self.Q2,
            self.R2
        )

        assert isinstance(P, np.ndarray)
        assert isinstance(info, dict)

    def test_info_contains_expected_fields(self):

        _, info = solve_dare(
            self.A2,
            self.B2,
            self.Q2,
            self.R2
        )

        expected_fields = {
            "converged",
            "iterations",
            "error"
        }

        assert expected_fields.issubset(info.keys())

    # ======================================================
    # CONVERGENCE
    # ======================================================

    def test_converges(self):

        _, info = solve_dare(
            self.A2,
            self.B2,
            self.Q2,
            self.R2
        )

        assert info["converged"], (
            f"DARE did not converge "
            f"(error={info['error']:.2e})"
        )

    def test_iteration_count_recorded(self):

        _, info = solve_dare(
            self.A2,
            self.B2,
            self.Q2,
            self.R2
        )

        assert info["iterations"] > 0

    def test_no_convergence_with_one_iteration(self):

        _, info = solve_dare(
            self.A2,
            self.B2,
            self.Q2,
            self.R2,
            max_iter=1
        )

        assert not info["converged"]

    # ======================================================
    # MATRIX PROPERTIES
    # ======================================================

    def test_output_shape(self):

        P, _ = solve_dare(
            self.A2,
            self.B2,
            self.Q2,
            self.R2
        )

        assert P.shape == (2, 2)

    def test_solution_is_symmetric(self):

        P, _ = solve_dare(
            self.A2,
            self.B2,
            self.Q2,
            self.R2
        )

        np.testing.assert_allclose(
            P,
            P.T,
            atol=1e-10,
            err_msg="P is not symmetric"
        )

    def test_solution_positive_definite(self):

        P, _ = solve_dare(
            self.A2,
            self.B2,
            self.Q2,
            self.R2
        )

        eigvals = np.linalg.eigvalsh(P)

        assert np.all(eigvals > 0), (
            f"P is not positive definite. "
            f"Eigenvalues: {eigvals}"
        )

    def test_solution_is_finite(self):

        P, _ = solve_dare(
            self.A2,
            self.B2,
            self.Q2,
            self.R2
        )

        assert np.all(np.isfinite(P))

    # ======================================================
    # BELLMAN EQUATION
    # ======================================================

    def test_bellman_residual(self):
        """
        Verify Riccati fixed-point equation.
        """

        P, _ = solve_dare(
            self.A2,
            self.B2,
            self.Q2,
            self.R2
        )

        gain = np.linalg.inv(
            self.R2 + self.B2.T @ P @ self.B2
        )

        P_check = (
            self.Q2
            + self.A2.T @ P @ self.A2
            - self.A2.T
            @ P
            @ self.B2
            @ gain
            @ self.B2.T
            @ P
            @ self.A2
        )

        np.testing.assert_allclose(
            P,
            P_check,
            atol=1e-6,
            err_msg="Bellman residual too large"
        )

    # ======================================================
    # CLOSED-LOOP STABILITY
    # ======================================================

    def test_closed_loop_stable(self):

        P, _ = solve_dare(
            self.A2,
            self.B2,
            self.Q2,
            self.R2
        )

        K = (
            np.linalg.inv(
                self.R2 + self.B2.T @ P @ self.B2
            )
            @ self.B2.T
            @ P
            @ self.A2
        )

        A_cl = self.A2 - self.B2 @ K

        eigvals = np.abs(
            np.linalg.eigvals(A_cl)
        )

        assert np.all(eigvals < 1.0), (
            f"Closed-loop unstable. "
            f"|λ| = {eigvals}"
        )

    # ======================================================
    # REALISTIC 4x4 SYSTEM
    # ======================================================

    def test_4x4_system_converges(self):

        A = np.diag([
            0.99,
            0.98,
            0.97,
            0.96
        ])

        B = np.ones((4, 1)) * 0.1

        Q = np.eye(4)

        R = np.array([
            [1.0]
        ])

        P, info = solve_dare(A, B, Q, R)

        assert info["converged"]

        assert P.shape == (4, 4)

        eigvals = np.linalg.eigvalsh(P)

        assert np.all(eigvals > 0)

    # ======================================================
    # EDGE CASES
    # ======================================================

    def test_singular_R_behaviour(self):

        R_bad = np.array([[0.0]])

        (P, info) = solve_dare(
            self.A2,
            self.B2,
            self.Q2,
            R_bad
        )

        # still must return something valid
        assert np.all(np.isfinite(P))

        # may or may not converge depending on system
        assert P.shape == (2, 2)

    def test_large_Q_still_converges(self):

        Q_large = np.eye(2) * 1e6

        P, info = solve_dare(
            self.A2,
            self.B2,
            Q_large,
            self.R2
        )

        assert info["converged"]

        assert np.all(np.isfinite(P))

    def test_small_R_still_converges(self):

        R_small = np.array([
            [1e-6]
        ])

        P, info = solve_dare(
            self.A2,
            self.B2,
            self.Q2,
            R_small
        )

        assert info["converged"]

        assert np.all(np.isfinite(P))

    # ======================================================
    # INVALID INPUTS
    # ======================================================

    def test_invalid_Q_dimension_raises(self):

        Q_bad = np.eye(3)

        with pytest.raises(ValueError):

            solve_dare(
                self.A2,
                self.B2,
                Q_bad,
                self.R2
            )

    def test_invalid_R_dimension_raises(self):

        R_bad = np.eye(2)

        with pytest.raises(ValueError):

            solve_dare(
                self.A2,
                self.B2,
                self.Q2,
                R_bad
            )

    def test_non_square_A_raises(self):

        A_bad = np.ones((2, 3))

        with pytest.raises(ValueError):

            solve_dare(
                A_bad,
                self.B2,
                self.Q2,
                self.R2
            )