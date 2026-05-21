import numpy as np
from core.math_utils import solve_dare


class LQRController:

    def __init__(self, A, B, Q, R):

        self.A = np.asarray(A)
        self.B = np.asarray(B)
        self.Q = np.asarray(Q)
        self.R = np.asarray(R)

        # HARD SAFETY: force 2D
        self.A = np.atleast_2d(self.A)
        self.B = np.atleast_2d(self.B)
        self.Q = np.atleast_2d(self.Q)

        # IMPORTANT FIX
        self.R = np.atleast_2d(self.R)

        self.K = self._compute_gain()

    def _compute_gain(self):

        P, _ = solve_dare(self.A, self.B, self.Q, self.R)

        P = np.asarray(P)

        S = self.R + self.B.T @ P @ self.B

        K = np.linalg.inv(S) @ (self.B.T @ P @ self.A)

        return K

    def control(self, x):
        x = np.asarray(x).reshape(-1, 1)
        return -self.K @ x