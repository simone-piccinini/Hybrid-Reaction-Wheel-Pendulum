import numpy as np
from scipy.linalg import solve_discrete_are, inv
from scipy.signal import cont2discrete



def solve_dare(A, B, Q, R, max_iter=2000, tol=1e-10):
        """Semplice risolutore DARE tramite iterazione della matrice di costo P."""
        P = Q.copy()
        info = {'converged': False, 'iterations': 0, 'error': 0.0}
        
        for i in range(max_iter):
            # Termine di guadagno intermedio per leggibilità e velocità
            # G = inv(R + B.T @ P @ B) @ (B.T @ P @ A)
            knl = np.linalg.inv(R + B.T @ P @ B)
            P_new = Q + A.T @ P @ A - (A.T @ P @ B @ knl @ B.T @ P @ A)
            
            err = np.max(np.abs(P_new - P))
            if err < tol:
                info['converged'] = True
                info['iterations'] = i + 1
                info['error'] = err
                return P_new, info
            P = P_new
            
        info['error'] = err
        info['iterations'] = max_iter
        return P, info

class LQGController:
    def __init__(self, config, wheel, motor, dt):
        self.cfg = config
        self.dt = dt
        
        m = config.pendulum.m
        l = config.pendulum.l / 2  # distance from center of mass
        g = config.g
        Ip = (1/3) * m * (config.pendulum.l**2) 
        Iw = wheel.inertia
        bp = config.pendulum.b
        bw = wheel.b
        
        # Constants
        E = motor.K_t / motor.R_a
        D = (motor.K_t * motor.K_e / motor.R_a) + bw

        #Linear system around equilibrium point
        A_c = np.array([
            [0, 1, 0, 0],
            [(m * g * l) / Ip, -(bp / Ip), 0, D / Ip],
            [0, 0, 0, 1],
            [-(m * g * l) / Ip, (bp / Ip), 0, -(D * (Ip + Iw)) / (Ip * Iw)],
        ])

        B_c = np.array([
            [0],
            [-E / Ip],
            [0],
            [(E * (Ip + Iw)) / (Ip * Iw)]
        ])

        C_c = np.array([
            [1, 0, 0, 0], # Misura Theta (encoder 1)
            [0, 0, 0, 1]  # Misura Phi_dot (velocità ruota)
        ])
        D_c = np.zeros((2, 1))

        # Discretization (ZOH)
        self.A_d, self.B_d, self.C_d, self.D_d, _ = cont2discrete((A_c, B_c, C_c, D_c), dt)

        ### MATRIX of HYPERPARAMETERS

        ## matrix that contains my hyperparameters
        self.Q_lqr = np.diag([1,1,1,1]) # 4 x 4 equal to the number of states
        self.R_lqr = np.array([[1.0]]) # scalar because there is a single motor

        self.W_kf = np.diag([1,1,1,1]) # 4 x 4 process noise covariance 
        self.V_kf = np.diag([1,1]) # measurement noise covariance 2 x 2 two sensors

        # LQR gain
        P_lqr, info_lqr = solve_dare(self.A_d, self.B_d, self.Q_lqr, self.R_lqr)
        self.K_lqr = np.linalg.inv(self.R_lqr + self.B_d.T @ P_lqr @ self.B_d) @ self.B_d.T @ P_lqr @ self.A_d
        
        # Kalman gain
        P_kf, info_kf = solve_dare(self.A_d.T, self.C_d.T, self.W_kf, self.V_kf)
        self.L = P_kf @ self.C_d.T @ np.linalg.inv(self.C_d @ P_kf @ self.C_d.T + self.V_kf)
        
       
        # Inizializzazione stato stimato
        self.x_hat = np.zeros((4, 1))


    def compute_action(self, y_meas, u_last):

        innovation = y_meas - (self.C_d @ self.x_hat)
        self.x_hat = self.x_hat + self.L @ innovation

        # 2. Legge di controllo (LQR)
        V_a = -self.K_lqr @ self.x_hat
        V_a_sat = np.clip(V_a[0, 0], -12, 12) # Saturazione hardware

        # 3. Predizione per il prossimo step
        # x_hat(k+1) = A * x_hat(k) + B * u(k)
        self.x_hat = self.A_d @ self.x_hat + self.B_d * V_a_sat

        return V_a_sat