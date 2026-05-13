class KalmanFilter:

    def __init__(self, A, B, C, W, V):

        self.A = np.atleast_2d(A).astype(float)
        self.B = np.atleast_2d(B).astype(float)
        self.C = np.atleast_2d(C).astype(float)

        self.W = np.atleast_2d(W).astype(float)
        self.V = np.atleast_2d(V).astype(float)

        self.L = self._compute_gain()

        n = self.A.shape[0]
        self.x_hat = np.zeros((n, 1), dtype=float)

    def _compute_gain(self):

        P, _ = solve_dare(
            self.A.T,
            self.C.T,
            self.W,
            self.V
        )

        P = np.atleast_2d(P)

        S = self.C @ P @ self.C.T + self.V
        L = P @ self.C.T @ np.linalg.inv(S)

        return np.atleast_2d(L)

    def predict(self, u):
        u = np.atleast_2d(u)
        self.x_hat = self.A @ self.x_hat + self.B @ u

    def update(self, y):
        y = np.atleast_2d(y)
        y_hat = self.C @ self.x_hat
        self.x_hat = self.x_hat + self.L @ (y - y_hat)

    def estimate(self, y, u=None):
        self.predict(u if u is not None else np.zeros((1,1)))
        self.update(y)
        return self.x_hat