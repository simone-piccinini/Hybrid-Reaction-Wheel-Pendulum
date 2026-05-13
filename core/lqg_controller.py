class LQGController:

    def __init__(
        self,
        lqr_controller,
        kalman_filter
    ):

        self.lqr = lqr_controller
        self.kf = kalman_filter

    def compute_action(self, y_meas):

        x_hat = self.kf.x_hat

        u = self.lqr.control(x_hat)

        self.kf.estimate(y_meas, u)

        return u