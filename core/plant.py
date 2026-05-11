import numpy as np

class ReactionWheelPendulum:
    def __init__(self, config, wheel, motor):
        self.cfg = config
        self.wheel = wheel
        self.motor = motor # Iniettiamo il motore
        self.state = np.zeros(4) 
        self.I_p = (1/3) * self.cfg.pendulum.m * (self.cfg.pendulum.l**2)

    def dynamics(self, x, V_a):
        """
        x = [theta, theta_dot, phi, phi_dot]
        V_a = Tensione applicata al motore (Input di controllo)
        """
        theta, th_dot, phi, ph_dot = x
        
        # Parametri
        m = self.cfg.pendulum.m
        l = self.cfg.pendulum.l
        g = self.cfg.g
        Ip = self.I_p
        Iw = self.wheel.inertia
        bp = self.cfg.pendulum.b
        bw = self.wheel.b

        # Compute torque of motor pure.
        u_motor = self.motor.get_torque(V_a, ph_dot)

        # Lagrangian equations of motion
        # (Ip + Iw)th_ddot + (u_motor - bw*ph_dot - Iw*th_ddot) - mgl*sin(theta) = -bp*th_dot
        # Ip*th_ddot = mgl*sin(theta) - bp*th_dot - (u_motor - bw*ph_dot)
        
        #put equations instandard matrix form
        
        M = np.array([[Ip + Iw, Iw],
                    [Iw,      Iw]])

        F = np.array([m*g*(l/2)*np.sin(theta) - bp*th_dot,
                    u_motor - bw*ph_dot])

        # Solving system M * q_ddot = F  =>  q_ddot = M^-1 * F
        q_ddot = np.linalg.solve(M, F)

        th_ddot = q_ddot[0]
        ph_ddot = q_ddot[1]

        return np.array([th_dot, th_ddot, ph_dot, ph_ddot])