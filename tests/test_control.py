"""
test_control.py
---------------
pytest suite for control.py — LQG controller for an inverted wheeled pendulum.

Run with:
    pytest test_control.py -v

Covers:
    - solve_dare            : convergence, symmetry, positive-definiteness, Bellman residual
    - LQGController.__init__: matrix shapes, stability of closed-loop, Kalman gain properties
    - LQGController.compute_action: state update, output saturation, innovation step
"""

import numpy as np
import pytest
from types import SimpleNamespace

# ── import the module under test ──────────────────────────────────────────────

from core.control import LQGController, solve_dare

# ══════════════════════════════════════════════════════════════════════════════
# Helpers / fixtures
# ══════════════════════════════════════════════════════════════════════════════

def make_config(
    m=0.5, l=0.3, g=9.81, b_pend=0.01
):
    """Return a SimpleNamespace that mimics the real Config object."""
    pendulum = SimpleNamespace(m=m, l=l, b=b_pend)
    return SimpleNamespace(pendulum=pendulum, g=g)


def make_wheel(inertia=1e-4, b=1e-4):
    return SimpleNamespace(inertia=inertia, b=b)


def make_motor(K_t=0.3, K_e=0.3, R_a=5.0):
    return SimpleNamespace(K_t=K_t, K_e=K_e, R_a=R_a)


@pytest.fixture
def default_controller():
    """A fully-constructed LQGController with physically plausible parameters."""
    cfg   = make_config()
    wheel = make_wheel()
    motor = make_motor()
    dt    = 0.01
    return LQGController(cfg, wheel, motor, dt)


# ══════════════════════════════════════════════════════════════════════════════
# 1. solve_dare
# ══════════════════════════════════════════════════════════════════════════════

class TestSolveDare:
    """Unit tests for the standalone DARE solver."""

    # Minimal stabilisable 2×2 system
    A2 = np.array([[1.1, 0.0],
                   [0.0, 0.9]])
    B2 = np.array([[1.0],
                   [1.0]])
    Q2 = np.eye(2)
    R2 = np.array([[1.0]])

    def test_returns_two_values(self):
        result = solve_dare(self.A2, self.B2, self.Q2, self.R2)
        assert len(result) == 2

    def test_converges(self):
        _, info = solve_dare(self.A2, self.B2, self.Q2, self.R2)
        assert info['converged'], (
            f"DARE did not converge (error={info['error']:.2e})"
        )

    def test_output_shape(self):
        P, _ = solve_dare(self.A2, self.B2, self.Q2, self.R2)
        assert P.shape == (2, 2)

    def test_positive_definite(self):
        P, _ = solve_dare(self.A2, self.B2, self.Q2, self.R2)
        eigvals = np.linalg.eigvalsh(P)
        assert np.all(eigvals > 0), f"P is not positive definite: eigenvalues={eigvals}"

    def test_symmetric(self):
        P, _ = solve_dare(self.A2, self.B2, self.Q2, self.R2)
        np.testing.assert_allclose(P, P.T, atol=1e-10,
                                   err_msg="Solution P is not symmetric")

    def test_bellman_residual(self):
        """P must satisfy the Bellman equation to within tolerance."""
        P, _ = solve_dare(self.A2, self.B2, self.Q2, self.R2)
        # DARE: P = Q + A'PA - A'PB(R+B'PB)^{-1}B'PA
        gain = np.linalg.inv(self.R2 + self.B2.T @ P @ self.B2)
        P_check = (self.Q2
                   + self.A2.T @ P @ self.A2
                   - self.A2.T @ P @ self.B2 @ gain @ self.B2.T @ P @ self.A2)
        np.testing.assert_allclose(P, P_check, atol=1e-6,
                                   err_msg="DARE Bellman residual too large")

    def test_iteration_count_recorded(self):
        _, info = solve_dare(self.A2, self.B2, self.Q2, self.R2)
        assert info['iterations'] > 0

    def test_no_convergence_within_one_iteration(self):
        """With max_iter=1 on a non-trivial system the solver should NOT converge."""
        _, info = solve_dare(self.A2, self.B2, self.Q2, self.R2, max_iter=1)
        assert not info['converged']

    def test_4x4_system_converges(self):
        """Same dimensionality as the real plant (4-state, 1-input)."""
        A = np.diag([0.99, 0.98, 0.97, 0.96])
        B = np.ones((4, 1)) * 0.1
        Q = np.eye(4)
        R = np.array([[1.0]])
        P, info = solve_dare(A, B, Q, R)
        assert info['converged']
        assert P.shape == (4, 4)
        eigvals = np.linalg.eigvalsh(P)
        assert np.all(eigvals > 0)


# ══════════════════════════════════════════════════════════════════════════════
# 2. LQGController — initialisation
# ══════════════════════════════════════════════════════════════════════════════

class TestLQGControllerInit:

    # ── discretised system matrices ──────────────────────────────────────────

    def test_A_d_shape(self, default_controller):
        assert default_controller.A_d.shape == (4, 4)

    def test_B_d_shape(self, default_controller):
        assert default_controller.B_d.shape == (4, 1)

    def test_C_d_shape(self, default_controller):
        assert default_controller.C_d.shape == (2, 4)

    def test_D_d_shape(self, default_controller):
        assert default_controller.D_d.shape == (2, 1)

    def test_A_d_is_finite(self, default_controller):
        assert np.all(np.isfinite(default_controller.A_d))

    def test_B_d_is_finite(self, default_controller):
        assert np.all(np.isfinite(default_controller.B_d))

    # ── LQR gain ─────────────────────────────────────────────────────────────

    def test_K_lqr_shape(self, default_controller):
        assert default_controller.K_lqr.shape == (1, 4), (
            "K_lqr should be (1, 4) for a single-input, 4-state system"
        )

    def test_K_lqr_is_finite(self, default_controller):
        assert np.all(np.isfinite(default_controller.K_lqr))

    def test_closed_loop_stable(self, default_controller):
        """All eigenvalues of (A_d - B_d K_lqr) must lie strictly inside unit circle."""
        ctrl = default_controller
        A_cl = ctrl.A_d - ctrl.B_d @ ctrl.K_lqr
        eigvals = np.abs(np.linalg.eigvals(A_cl))
        assert np.all(eigvals < 1.0), (
            f"Closed-loop is unstable. |λ| = {eigvals}"
        )

    # ── Kalman gain ───────────────────────────────────────────────────────────

    def test_L_shape(self, default_controller):
        assert default_controller.L.shape == (4, 2), (
            "L should be (4, 2): 4 states × 2 measurements"
        )

    def test_L_is_finite(self, default_controller):
        assert np.all(np.isfinite(default_controller.L))

    #notes on this important:
    #Passes for the expected marginal z = 1 pole
    # Fails loudly if any pole goes strictly outside the unit circle
    # if someone accidentally makes φ observable later, the marginal == 1 assertion will catch the change
    def test_observer_stable(self, default_controller):
        ctrl = default_controller
        A_obs = ctrl.A_d - ctrl.L @ ctrl.C_d
        eigvals = np.abs(np.linalg.eigvals(A_obs))

        # φ is unobservable (no sensor measures it, it doesn't influence
        # other states) → one pole sits exactly at z=1. This is expected
        # and physically correct; only strictly-outside-unit-circle is a bug.
        assert not np.any(eigvals > 1.0 + 1e-9), (
            f"Observer has unstable pole(s). |λ| = {eigvals}"
        )

        # Verify exactly one marginal pole (the φ integrator)
        marginal = np.sum(np.abs(eigvals - 1.0) < 1e-9)
        assert marginal == 1, (
            f"Expected exactly 1 marginal pole (φ unobservable), got {marginal}. |λ| = {eigvals}"
    )

    # ── initial state ─────────────────────────────────────────────────────────

    def test_x_hat_initial_zeros(self, default_controller):
        np.testing.assert_array_equal(
            default_controller.x_hat, np.zeros((4, 1))
        )

    def test_x_hat_shape(self, default_controller):
        assert default_controller.x_hat.shape == (4, 1)

    # ── hyperparameter matrices ───────────────────────────────────────────────

    def test_Q_lqr_shape(self, default_controller):
        assert default_controller.Q_lqr.shape == (4, 4)

    def test_R_lqr_shape(self, default_controller):
        assert default_controller.R_lqr.shape == (1, 1)

    def test_W_kf_shape(self, default_controller):
        assert default_controller.W_kf.shape == (4, 4)

    def test_V_kf_shape(self, default_controller):
        assert default_controller.V_kf.shape == (2, 2)

    # ── dt stored ─────────────────────────────────────────────────────────────

    def test_dt_stored(self):
        dt = 0.005
        ctrl = LQGController(make_config(), make_wheel(), make_motor(), dt)
        assert ctrl.dt == dt

    # ── different dt produces different discretisation ────────────────────────

    def test_different_dt_gives_different_A_d(self):
        cfg, whl, mtr = make_config(), make_wheel(), make_motor()
        ctrl1 = LQGController(cfg, whl, mtr, dt=0.01)
        ctrl2 = LQGController(cfg, whl, mtr, dt=0.05)
        assert not np.allclose(ctrl1.A_d, ctrl2.A_d)


# ══════════════════════════════════════════════════════════════════════════════
# 3. LQGController — compute_action
# ══════════════════════════════════════════════════════════════════════════════

class TestComputeAction:
    """
    NOTE: the original code in compute_action contains attribute-name bugs
    (self.C / self.K / self.B instead of self.C_d / self.K_lqr / self.B_d).
    The tests below document the *correct* expected behaviour and will FAIL
    until those bugs are fixed, acting as regression guards.
    """

    def test_returns_scalar(self, default_controller):
        y = np.zeros((2, 1))
        u_last = 0.0
        result = default_controller.compute_action(y, u_last)
        assert np.isscalar(result) or result.shape == ()

    def test_zero_state_zero_output(self, default_controller):
        """With zero state and zero measurement the control action must be 0."""
        default_controller.x_hat = np.zeros((4, 1))
        y = np.zeros((2, 1))
        Va = default_controller.compute_action(y, 0.0)
        assert Va == pytest.approx(0.0, abs=1e-10)

    def test_saturation_positive(self, default_controller):
        """Large positive state → output must not exceed +12 V."""
        default_controller.x_hat = np.ones((4, 1)) * 1e6
        y = np.zeros((2, 1))
        Va = default_controller.compute_action(y, 0.0)
        assert Va <= 12.0

    def test_saturation_negative(self, default_controller):
        """Large negative state → output must not go below −12 V."""
        default_controller.x_hat = -np.ones((4, 1)) * 1e6
        y = np.zeros((2, 1))
        Va = default_controller.compute_action(y, 0.0)
        assert Va >= -12.0

    def test_output_within_bounds_random(self, default_controller):
        """Output is always within hardware saturation for random measurements."""
        rng = np.random.default_rng(42)
        for _ in range(50):
            y = rng.uniform(-5, 5, size=(2, 1))
            Va = default_controller.compute_action(y, 0.0)
            assert -12.0 <= Va <= 12.0

    def test_x_hat_updates_after_call(self, default_controller):
        """x_hat must change after a compute_action call (prediction step)."""
        x_hat_before = default_controller.x_hat.copy()
        y = np.array([[0.1], [0.0]])
        default_controller.compute_action(y, 0.0)
        assert not np.allclose(default_controller.x_hat, x_hat_before), (
            "x_hat was not updated after compute_action"
        )

    def test_x_hat_shape_preserved(self, default_controller):
        """x_hat must remain (4, 1) after each call."""
        y = np.array([[0.05], [0.01]])
        for _ in range(10):
            default_controller.compute_action(y, 0.0)
        assert default_controller.x_hat.shape == (4, 1)

    def test_x_hat_finite_after_many_steps(self, default_controller):
        """Closed-loop observer must not diverge over 500 steps at small angle."""
        rng = np.random.default_rng(0)
        default_controller.x_hat = np.array([[0.05], [0.0], [0.0], [0.0]])
        for _ in range(500):
            noise = rng.normal(0, 1e-3, size=(2, 1))
            y = default_controller.C_d @ default_controller.x_hat + noise
            default_controller.compute_action(y, 0.0)
        assert np.all(np.isfinite(default_controller.x_hat)), (
            "x_hat diverged over 500 steps"
        )

    def test_innovation_uses_C_d(self, default_controller):
        """
        Regression: original code uses self.C (undefined) instead of self.C_d.
        This test ensures the correct attribute is used (no AttributeError).
        """
        default_controller.x_hat = np.array([[0.1], [0.0], [0.0], [0.0]])
        y = default_controller.C_d @ default_controller.x_hat
        # Should NOT raise AttributeError
        default_controller.compute_action(y, 0.0)

    def test_control_uses_K_lqr(self, default_controller):
        """
        Regression: original code uses self.K (undefined) instead of self.K_lqr.
        Verified implicitly — if the attribute is wrong an AttributeError is raised.
        """
        y = np.zeros((2, 1))
        default_controller.compute_action(y, 0.0)   # must not raise

    def test_prediction_uses_B_d(self, default_controller):
        """
        Regression: original code uses self.B (undefined) instead of self.B_d.
        Verified implicitly via the absence of AttributeError.
        """
        y = np.zeros((2, 1))
        for _ in range(5):
            default_controller.compute_action(y, 0.0)   # must not raise


# ══════════════════════════════════════════════════════════════════════════════
# 4. Edge-case / robustness
# ══════════════════════════════════════════════════════════════════════════════

class TestEdgeCases:

    def test_very_small_dt(self):
        """Controller must initialise without error for very small dt."""
        ctrl = LQGController(make_config(), make_wheel(), make_motor(), dt=1e-4)
        assert ctrl.A_d.shape == (4, 4)

    def test_very_large_dt(self):
        """Controller must initialise without error for a large dt (e.g. 0.5 s)."""
        ctrl = LQGController(make_config(), make_wheel(), make_motor(), dt=0.5)
        assert ctrl.A_d.shape == (4, 4)

    def test_multiple_compute_action_calls_stable(self):
        """100 sequential calls on the same controller must not raise."""
        ctrl = LQGController(make_config(), make_wheel(), make_motor(), dt=0.01)
        y = np.zeros((2, 1))
        for _ in range(100):
            ctrl.compute_action(y, 0.0)

    def test_reset_x_hat(self, default_controller):
        """Manually resetting x_hat mid-flight should work correctly."""
        y = np.ones((2, 1)) * 0.1
        for _ in range(20):
            default_controller.compute_action(y, 0.0)
        default_controller.x_hat = np.zeros((4, 1))
        Va = default_controller.compute_action(np.zeros((2, 1)), 0.0)
        assert Va == pytest.approx(0.0, abs=1e-10)