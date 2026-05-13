import numpy as np
import pytest
from types import SimpleNamespace

from core.parameters_space import ParameterSpace


# ======================================================
# FIXTURES
# ======================================================

def make_config():
    pendulum = SimpleNamespace(
        m=0.5,
        l=0.3,
        b=0.01
    )
    return SimpleNamespace(pendulum=pendulum, g=9.81)


def make_wheel():
    return SimpleNamespace(inertia=1e-4, b=1e-4)


def make_motor():
    return SimpleNamespace(
        K_t=0.3,
        K_e=0.3,
        R_a=5.0
    )


@pytest.fixture
def params():
    return ParameterSpace(
        config=make_config(),
        wheel=make_wheel(),
        motor=make_motor(),
        dt=0.01
    )


# ======================================================
# BASIC STRUCTURE TESTS
# ======================================================

class TestParameterSpaceInit:

    def test_A_shape(self, params):
        assert params.A.shape == (4, 4)

    def test_B_shape(self, params):
        assert params.B.shape == (4, 1)

    def test_C_shape(self, params):
        assert params.C.shape == (2, 4)

    def test_D_shape(self, params):
        assert params.D.shape == (2, 1)

    def test_Q_shape(self, params):
        assert params.Q.shape == (4, 4)

    def test_R_shape(self, params):
        assert params.R.shape == (1, 1)

    def test_W_shape(self, params):
        assert params.W.shape == (4, 4)

    def test_V_shape(self, params):
        assert params.V.shape == (2, 2)

    def test_all_finite(self, params):
        for M in [params.A, params.B, params.C, params.D, params.Q, params.R, params.W, params.V]:
            assert np.all(np.isfinite(M))


# ======================================================
# SETTERS SAFETY TESTS
# ======================================================

class TestParameterSpaceSetters:

    def test_set_lqr_weights_preserves_shape(self, params):

        Q_new = np.eye(4) * 10
        R_new = np.array([0.5])   # intentionally wrong shape

        params.set_lqr_weights(Q_new, R_new)

        assert params.Q.shape == (4, 4)
        assert params.R.shape == (1, 1)

    def test_set_kalman_covariances_preserves_shape(self, params):

        W_new = np.eye(4) * 0.1
        V_new = np.array([0.2, 0.2])  # intentionally wrong shape

        params.set_kalman_covariances(W_new, V_new)

        assert params.W.shape == (4, 4)
        assert params.V.shape == (2, 2)

    def test_setter_does_not_change_values_significantly(self, params):

        old_Q = params.Q.copy()

        Q_new = np.eye(4) * 2.0
        params.set_lqr_weights(Q_new, np.array([[1.0]]))

        assert not np.allclose(old_Q, params.Q)


# ======================================================
# INVARIANCE / CONSISTENCY TESTS
# ======================================================

class TestParameterSpaceConsistency:

    def test_discrete_system_shapes(self, params):
        assert params.A.shape[0] == params.A.shape[1]
        assert params.B.shape[0] == params.A.shape[0]
        assert params.C.shape[1] == params.A.shape[1]

    def test_Q_R_W_V_consistency(self, params):
        n = params.A.shape[0]
        m = params.B.shape[1]
        p = params.C.shape[0]

        assert params.Q.shape == (n, n)
        assert params.W.shape == (n, n)
        assert params.R.shape == (m, m)
        assert params.V.shape == (p, p)


# ======================================================
# ROBUSTNESS TESTS
# ======================================================

class TestParameterSpaceRobustness:

    def test_multiple_reassignments(self, params):

        for _ in range(10):
            params.set_lqr_weights(
                np.eye(4),
                np.array([[1.0]])
            )

        assert params.Q.shape == (4, 4)
        assert params.R.shape == (1, 1)

    def test_no_nan_after_operations(self, params):

        params.set_lqr_weights(
            np.eye(4) * 5,
            np.array([[0.1]])
        )

        params.set_kalman_covariances(
            np.eye(4),
            np.eye(2)
        )

        for M in [params.Q, params.R, params.W, params.V]:
            assert np.all(np.isfinite(M))