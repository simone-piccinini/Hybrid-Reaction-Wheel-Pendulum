import numpy as np
import pytest

from core.parameters_space import ParameterSpace
from core.kalman_filter import KalmanFilter

#pytest tests/test_kalman_filter.py -v

# ======================================================
# FIXTURE: realistic system from your plant structure
# ======================================================

@pytest.fixture
def params():
    from core.config_loader import PhysicalConfig
    from core.components import ReactionWheel, DCMotor

    cfg = PhysicalConfig.from_yaml("config/params.yaml")

    wheel = ReactionWheel(cfg.wheel.m, cfg.wheel.r, cfg.wheel.b)
    motor = DCMotor(cfg.motor.K_t, cfg.motor.K_e, cfg.motor.R_a)

    return ParameterSpace(cfg, wheel, motor, dt=0.01)


@pytest.fixture
def kf(params):
    return KalmanFilter(
        params.A,
        params.B,
        params.C,
        params.W,
        params.V
    )


# ======================================================
# 1. INITIALISATION CONTRACT
# ======================================================

class TestKalmanFilterInit:

    def test_A_shape(self, kf):
        assert kf.A.shape == (4, 4)

    def test_B_shape(self, kf):
        assert kf.B.shape == (4, 1)

    def test_C_shape(self, kf):
        assert kf.C.shape == (2, 4)

    def test_W_shape(self, kf):
        assert kf.W.shape == (4, 4)

    def test_V_shape(self, kf):
        assert kf.V.shape == (2, 2)

    def test_x_hat_initial_zero(self, kf):
        assert np.allclose(kf.x_hat, np.zeros((4, 1)))

    def test_x_hat_shape(self, kf):
        assert kf.x_hat.shape == (4, 1)

    def test_finite_matrices(self, kf):
        assert np.all(np.isfinite(kf.A))
        assert np.all(np.isfinite(kf.C))


# ======================================================
# 2. KALMAN GAIN PROPERTIES
# ======================================================

class TestKalmanGain:

    def test_gain_exists(self, kf):
        assert hasattr(kf, "L")

    def test_gain_shape(self, kf):
        assert kf.L.shape == (4, 2)

    def test_gain_finite(self, kf):
        assert np.all(np.isfinite(kf.L))


# ======================================================
# 3. OBSERVER STABILITY
# ======================================================

class TestObserverStability:

    def test_observer_stable(self, kf):

        A_obs = kf.A - kf.L @ kf.C
        eigvals = np.abs(np.linalg.eigvals(A_obs))

        assert np.all(eigvals <= 1.0 + 1e-9)

    def test_at_least_one_marginal_mode(self, kf):

        A_obs = kf.A - kf.L @ kf.C
        eigvals = np.abs(np.linalg.eigvals(A_obs))

        marginal = np.sum(np.isclose(eigvals, 1.0, atol=1e-6))

        assert marginal >= 1


# ======================================================
# 4. ESTIMATION DYNAMICS
# ======================================================

class TestKalmanEstimation:

    def test_state_updates(self, kf):

        y = np.array([[0.1], [0.0]])

        x_before = kf.x_hat.copy()

        kf.estimate(y)

        assert not np.allclose(kf.x_hat, x_before)

    def test_zero_measurement_behavior(self, kf):

        y = np.zeros((2, 1))

        kf.x_hat = np.zeros((4, 1))

        kf.estimate(y)

        assert np.all(np.isfinite(kf.x_hat))

    def test_shape_preserved(self, kf):

        y = np.array([[0.2], [0.1]])

        for _ in range(10):
            kf.estimate(y)

        assert kf.x_hat.shape == (4, 1)


# ======================================================
# 5. NUMERICAL ROBUSTNESS
# ======================================================

class TestKalmanRobustness:

    def test_noise_stability(self, kf):

        rng = np.random.default_rng(0)

        kf.x_hat = np.array([[0.1], [0.0], [0.0], [0.0]])

        for _ in range(200):

            noise = rng.normal(0, 1e-3, size=(2, 1))
            y = kf.C @ kf.x_hat + noise

            kf.estimate(y)

        assert np.all(np.isfinite(kf.x_hat))

    def test_large_measurement_does_not_explode(self, kf):

        y = np.ones((2, 1)) * 1e6

        kf.estimate(y)

        assert np.all(np.isfinite(kf.x_hat))


# ======================================================
# 6. INTEGRATION CONTRACT (lightweight)
# ======================================================

class TestKalmanIntegration:

    def test_prediction_step_consistency(self, kf):

        y = np.zeros((2, 1))

        kf.estimate(y)

        # ensures A, B, C are actually used coherently
        assert np.all(np.isfinite(kf.A @ kf.x_hat))