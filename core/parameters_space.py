import numpy as np
from scipy.signal import cont2discrete


class ParameterSpace:

    def __init__(self, config, wheel, motor, dt):

        self.dt = dt

        m = config.pendulum.m
        l = config.pendulum.l / 2
        g = config.g

        Ip = (1 / 3) * m * (config.pendulum.l ** 2)
        Iw = wheel.inertia

        bp = config.pendulum.b
        bw = wheel.b

        E = motor.K_t / motor.R_a
        D = (motor.K_t * motor.K_e / motor.R_a) + bw

        A_c = np.array([
            [0, 1, 0, 0],
            [(m * g * l) / Ip, -(bp / Ip), 0, D / Ip],
            [0, 0, 0, 1],
            [-(m * g * l) / Ip, (bp / Ip), 0,
             -(D * (Ip + Iw)) / (Ip * Iw)],
        ])

        B_c = np.array([
            [0],
            [-E / Ip],
            [0],
            [(E * (Ip + Iw)) / (Ip * Iw)]
        ])

        C_c = np.array([
            [1, 0, 0, 0],
            [0, 0, 0, 1]
        ])

        D_c = np.zeros((2, 1))

        self.A, self.B, self.C, self.D, _ = cont2discrete(
            (A_c, B_c, C_c, D_c),
            dt
        )

        n = self.A.shape[0]
        m_in = self.B.shape[1]
        p = self.C.shape[0]

        # safe defaults
        self.Q = np.eye(n)
        self.R = np.eye(m_in)
        self.W = np.eye(n)
        self.V = np.eye(p)

    # --------------------------
    # LQR
    # --------------------------

    def set_lqr_weights(self, Q, R):

        Q = np.asarray(Q)
        R = np.asarray(R)

        n = self.A.shape[0]
        m_in = self.B.shape[1]

        if Q.ndim == 1:
            Q = np.diag(Q)
        if R.ndim == 1:
            R = np.diag(R)

        if Q.shape != (n, n):
            raise ValueError(f"Q must be {(n, n)}, got {Q.shape}")

        if R.shape != (m_in, m_in):
            raise ValueError(f"R must be {(m_in, m_in)}, got {R.shape}")

        self.Q = Q
        self.R = R

    # --------------------------
    # Kalman
    # --------------------------

    def set_kalman_covariances(self, W, V):

        W = np.asarray(W)
        V = np.asarray(V)

        n = self.A.shape[0]
        p = self.C.shape[0]

        if W.ndim == 1:
            W = np.diag(W)
        if V.ndim == 1:
            V = np.diag(V)

        if W.shape != (n, n):
            raise ValueError(f"W must be {(n, n)}, got {W.shape}")

        if V.shape != (p, p):
            raise ValueError(f"V must be {(p, p)}, got {V.shape}")

        self.W = W
        self.V = V