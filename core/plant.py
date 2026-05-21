import numpy as np

class ReactionWheelPendulum:
    def __init__(self, config, wheel, motor):
        self.cfg = config
        self.wheel = wheel
        self.motor = motor # Iniettiamo il motore
        self.state = np.zeros(4) 
        self.I_p = (1/3) * self.cfg.pendulum.m * (self.cfg.pendulum.l**2)
    def dynamics(self, x, V_a):
        theta, th_dot, phi, ph_dot = x

        m = self.cfg.pendulum.m
        l = self.cfg.pendulum.l
        g = self.cfg.g
        Ip = self.I_p
        Iw = self.wheel.inertia
        bp = self.cfg.pendulum.b
        bw = self.wheel.b

        # force scalar (IMPORTANT)
        V_a = float(np.squeeze(V_a))

        u_motor = self.motor.get_torque(V_a, ph_dot)
        u_motor = float(np.squeeze(u_motor))

        M = np.array([
            [Ip + Iw, Iw],
            [Iw, Iw]
        ], dtype=float)

        F = np.array([
            m * g * (l / 2) * np.sin(theta) - bp * th_dot,
            u_motor - bw * ph_dot
        ], dtype=float)

        q_ddot = np.linalg.solve(M, F)

        return np.array([th_dot, q_ddot[0], ph_dot, q_ddot[1]], dtype=float)