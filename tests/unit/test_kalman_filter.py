"""Unit tests for estimation.kalman_filter (recursion, steady state, LQG loop)."""

import numpy as np
import pytest

from inverted_pendulum.core.types import LQGConfig, StateSpaceModel
from inverted_pendulum.dynamics.linearized_model import LinearizedPlantModel
from inverted_pendulum.dynamics.nonlinear_model import NonlinearPlantModel
from inverted_pendulum.control.lqr_controller import LQRController
from inverted_pendulum.estimation.kalman_filter import (
    KalmanFilter,
    steady_state_kalman_gain,
)
from inverted_pendulum.numerics.linalg import (
    NotPositiveDefiniteError,
    eigvals,
    is_symmetric,
)
from inverted_pendulum.numerics.riccati import RiccatiNotConverged
from inverted_pendulum.physical.motor import DCMotor
from inverted_pendulum.physical.pendulum import ReactionWheelPendulum
from inverted_pendulum.physical.wheel import ReactionWheel

DT = 0.01


def make_plant_model() -> LinearizedPlantModel:
    plant = ReactionWheelPendulum(
        pendulum_mass=0.3, pendulum_length=0.15, body_inertia=0.02,
        pivot_friction=0.01,
        wheel=ReactionWheel(mass=0.1, radius=0.05, inertia=1e-4,
                            friction_coefficient=1e-4),
        motor=DCMotor(resistance=2.0, torque_constant=0.05,
                      back_emf_constant=0.05),
    )
    return LinearizedPlantModel.from_plant(plant, DT)


def scalar_model(a=0.9, b=1.0, c=1.0) -> StateSpaceModel:
    return StateSpaceModel(
        [[a]], [[b]], [[c]], [[0.0]], is_discrete=True, dt=0.1
    )


def full_sensing_model() -> StateSpaceModel:
    m = make_plant_model()
    return StateSpaceModel(
        m.A_d, m.B_d, np.eye(4), np.zeros((4, 1)), is_discrete=True, dt=DT
    )


# --------------------------------------------------------------------------- #
# the recursion, hand-checked on a scalar system
# --------------------------------------------------------------------------- #
def test_scalar_predict():
    # x̂⁻ = a·x̂ + b·u ; P⁻ = a²P + W
    kf = KalmanFilter(scalar_model(), W_process=[[0.5]], V_measure=[[2.0]],
                      x0=[1.0], P0=[[4.0]])
    kf.predict([3.0])
    assert kf.x_hat[0] == pytest.approx(0.9 * 1.0 + 3.0)
    assert kf.P_est[0, 0] == pytest.approx(0.81 * 4.0 + 0.5)


def test_scalar_update():
    # S = P + V; L = P/S; x̂ += L(z − x̂); P = (1−L)²P + L²V (Joseph, scalar)
    kf = KalmanFilter(scalar_model(), W_process=[[0.5]], V_measure=[[2.0]],
                      x0=[1.0], P0=[[4.0]])
    L = 4.0 / 6.0
    kf.update([2.5])
    assert kf.x_hat[0] == pytest.approx(1.0 + L * (2.5 - 1.0))
    assert kf.P_est[0, 0] == pytest.approx((1 - L) ** 2 * 4.0 + L**2 * 2.0)


def test_zero_innovation_leaves_estimate():
    kf = KalmanFilter(scalar_model(), [[0.5]], [[2.0]], x0=[1.3], P0=[[4.0]])
    kf.update([1.3])  # z = C x̂ exactly
    assert kf.x_hat[0] == pytest.approx(1.3)


def test_predict_inflates_and_update_shrinks_uncertainty():
    m = make_plant_model()
    kf = KalmanFilter(m.discrete, np.eye(4), np.eye(2), P0=np.eye(4))
    before = np.trace(kf.P_est)
    kf.predict(np.zeros(1))
    inflated = np.trace(kf.P_est)
    kf.update(np.zeros(2))
    corrected = np.trace(kf.P_est)
    assert inflated > before
    assert corrected < inflated


def test_joseph_form_keeps_covariance_symmetric_psd():
    m = make_plant_model()
    kf = KalmanFilter(m.discrete, np.diag([1e-8, 1e-6, 1e-8, 1e-4]),
                      np.diag([1e-6, 1e-4]), P0=1e-2 * np.eye(4))
    rng = np.random.default_rng(3)
    for _ in range(100):
        kf.step(rng.standard_normal(1), rng.standard_normal(2))
        assert is_symmetric(kf.P_est)
        assert float(np.min(eigvals(kf.P_est).real)) >= -1e-12


def test_recursion_is_deterministic():
    m = make_plant_model()
    args = (m.discrete, np.eye(4) * 0.1, np.eye(2) * 0.2)
    a, b = KalmanFilter(*args), KalmanFilter(*args)
    rng = np.random.default_rng(7)
    for _ in range(20):
        u, z = rng.standard_normal(1), rng.standard_normal(2)
        np.testing.assert_array_equal(a.step(u, z), b.step(u, z))
    np.testing.assert_array_equal(a.P_est, b.P_est)


# --------------------------------------------------------------------------- #
# steady state — kalman.md's duality note
# --------------------------------------------------------------------------- #
def test_recursion_converges_to_dual_dare_steady_state():
    # detectable pair (full-state sensing), O(1) covariances so the §5 step
    # gate is dominated by its relative term
    model = full_sensing_model()
    W = np.diag([1.0, 2.0, 0.5, 3.0])
    V = 0.5 * np.eye(4)
    gain = steady_state_kalman_gain(model, W, V)
    kf = KalmanFilter(model, W, V, P0=np.eye(4))
    for _ in range(300):
        kf.predict(np.zeros(1))
        P_prior = kf.P_est.copy()
        kf.update(np.zeros(4))
    np.testing.assert_allclose(P_prior, gain.P_pred, rtol=1e-5)
    assert gain.L_gain.shape == (4, 4)


def test_steady_state_gain_formula():
    # L = P⁻Cᵀ(CP⁻Cᵀ + V)⁻¹ — check via the residual L(CP⁻Cᵀ+V) = P⁻Cᵀ
    model = full_sensing_model()
    W = np.diag([1.0, 2.0, 0.5, 3.0])
    V = 0.5 * np.eye(4)
    gain = steady_state_kalman_gain(model, W, V)
    S = model.C @ gain.P_pred @ model.C.T + V
    np.testing.assert_allclose(gain.L_gain @ S, gain.P_pred @ model.C.T,
                               rtol=1e-9, atol=1e-12)


def test_no_steady_state_for_the_unobservable_wheel_angle():
    # the locked sensor set [theta_p, theta_w_dot] leaves theta_w unobservable
    # at a unit-circle eigenvalue: the dual DARE has no stabilising solution
    # and the value iteration must refuse rather than fabricate one
    m = make_plant_model()
    W = np.diag([1e-8, 1e-6, 1e-8, 1e-4])
    V = np.diag([1e-6, 1e-4])
    with pytest.raises(RiccatiNotConverged):
        steady_state_kalman_gain(m.discrete, W, V)


def test_wheel_angle_variance_grows_without_bound():
    # the same physics seen by the recursion: P[2,2] only accumulates
    m = make_plant_model()
    kf = KalmanFilter(m.discrete, np.diag([1e-8, 1e-6, 1e-8, 1e-4]),
                      np.diag([1e-6, 1e-4]), P0=1e-2 * np.eye(4))
    samples = []
    for _ in range(200):
        kf.step(np.zeros(1), np.zeros(2))
        samples.append(kf.P_est[2, 2])
    diffs = np.diff(np.array(samples))
    assert np.all(diffs > 0.0)


# --------------------------------------------------------------------------- #
# the LQG closed loop — estimator + controller on the nonlinear plant
# --------------------------------------------------------------------------- #
def test_lqg_loop_stabilizes_the_nonlinear_plant_with_noisy_sensors():
    plant = ReactionWheelPendulum(
        pendulum_mass=0.3, pendulum_length=0.15, body_inertia=0.02,
        pivot_friction=0.01,
        wheel=ReactionWheel(mass=0.1, radius=0.05, inertia=1e-4,
                            friction_coefficient=1e-4),
        motor=DCMotor(resistance=2.0, torque_constant=0.05,
                      back_emf_constant=0.05),
    )
    linear = LinearizedPlantModel.from_plant(plant, DT)
    controller = LQRController.from_model(
        linear.discrete, np.diag([20.0, 2.0, 1e-2, 1e-2]), [[1.0]]
    )
    V_true = np.diag([1e-6, 1e-4])  # sensor noise: σ_θ = 1 mrad, σ_rate = 10 mrad/s
    kf = KalmanFilter(
        linear.discrete, np.diag([1e-8, 1e-6, 1e-8, 1e-4]), V_true,
        x0=np.zeros(4), P0=1e-2 * np.eye(4),
    )
    sim = NonlinearPlantModel(plant=plant, dt=DT)
    rng = np.random.default_rng(42)  # determinism: a single seeded generator
    noise_scale = np.sqrt(np.diag(V_true))

    x = np.array([0.05, 0.0, 0.0, 0.0])
    estimation_errors = []
    for _ in range(1000):
        u = controller.compute_control(kf.state_estimate)  # feedback on x̂
        x = sim.step(x, u[0])
        z = linear.C @ x + noise_scale * rng.standard_normal(2)
        kf.step(u, z)
        estimation_errors.append(abs(kf.x_hat[0] - x[0]))

    assert abs(x[0]) < 5e-3                       # held upright under noise
    assert abs(x[1]) < 5e-2
    assert np.mean(estimation_errors[500:]) < 1e-3  # θ̂ tracks θ within ~σ
    assert np.all(np.isfinite(kf.P_est))


# --------------------------------------------------------------------------- #
# guards / construction
# --------------------------------------------------------------------------- #
def test_rejects_continuous_model():
    m = make_plant_model()
    with pytest.raises(ValueError):
        KalmanFilter(m.continuous, np.eye(4), np.eye(2))
    with pytest.raises(ValueError):
        steady_state_kalman_gain(m.continuous, np.eye(4), np.eye(2))


def test_rejects_non_model():
    with pytest.raises(TypeError):
        KalmanFilter("not a model", np.eye(4), np.eye(2))


def test_rejects_malformed_noise():
    m = make_plant_model()
    with pytest.raises(ValueError):  # wrong shape
        KalmanFilter(m.discrete, np.eye(3), np.eye(2))
    with pytest.raises(ValueError):  # asymmetric W
        W = np.eye(4)
        W[0, 1] = 0.5
        KalmanFilter(m.discrete, W, np.eye(2))
    with pytest.raises(ValueError):  # W not PSD
        KalmanFilter(m.discrete, -np.eye(4), np.eye(2))
    with pytest.raises(NotPositiveDefiniteError):  # V only PSD, not PD
        KalmanFilter(m.discrete, np.eye(4), np.zeros((2, 2)))


def test_rejects_bad_u_and_z_shapes():
    kf = KalmanFilter(make_plant_model().discrete, np.eye(4), np.eye(2))
    with pytest.raises(ValueError):
        kf.predict(np.zeros(2))
    with pytest.raises(ValueError):
        kf.update(np.zeros(3))


def test_defaults_and_from_config():
    m = make_plant_model()
    W = np.diag([1.0, 2.0, 3.0, 4.0])
    kf = KalmanFilter(m.discrete, W, np.eye(2))
    np.testing.assert_array_equal(kf.x_hat, np.zeros(4))  # x0 default
    np.testing.assert_allclose(kf.P_est, W)               # P0 defaults to W

    config = LQGConfig(Q_lqr=np.eye(4), R_lqr=np.eye(1),
                       W_process=W, V_measure=2.0 * np.eye(2))
    from_config = KalmanFilter.from_config(m.discrete, config)
    np.testing.assert_allclose(from_config.W_process, W)
    np.testing.assert_allclose(from_config.V_measure, 2.0 * np.eye(2))
    with pytest.raises(TypeError):
        KalmanFilter.from_config(m.discrete, {"W": W})


def test_state_estimate_returns_a_copy():
    kf = KalmanFilter(make_plant_model().discrete, np.eye(4), np.eye(2))
    snapshot = kf.state_estimate
    snapshot[0] = 99.0
    assert kf.x_hat[0] == 0.0
