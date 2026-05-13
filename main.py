import numpy as np
import matplotlib.pyplot as plt

from core.config_loader import PhysicalConfig
from core.components import ReactionWheel, DCMotor
from core.plant import ReactionWheelPendulum
from core.simulation_engine import SimulationEngine

from core.parameters_space import ParameterSpace
from core.lqr_controller import LQRController
from core.kalman_filter import KalmanFilter


# ======================================================
# SIMPLE LQG (clean + explicit, no hidden bugs)
# ======================================================
class LQGController:

    def __init__(self, lqr, kf):
        self.lqr = lqr
        self.kf = kf

    def compute_action(self, y, u_prev):
        """
        LQG = Kalman state estimate + LQR feedback
        """

        # 1. estimate state
        x_hat = self.kf.estimate(y, u_prev)

        # 2. LQR control
        u = -self.lqr.K @ x_hat

        return u, x_hat


def main():

    # ======================================================
    # LOAD CONFIG
    # ======================================================
    config = PhysicalConfig.from_yaml("config/params.yaml")

    # ======================================================
    # PLANT
    # ======================================================
    wheel = ReactionWheel(
        mass=config.wheel.m,
        radius=config.wheel.r,
        friction=config.wheel.b
    )

    motor = DCMotor(
        K_t=config.motor.K_t,
        K_e=config.motor.K_e,
        R_a=config.motor.R_a
    )

    plant = ReactionWheelPendulum(config, wheel, motor)

    # ======================================================
    # SIMULATION
    # ======================================================
    dt = 0.01
    sim = SimulationEngine(plant=plant, dt=dt, t_end=5.0)

    # ======================================================
    # PARAMETERS (ALL MATRICES HERE)
    # ======================================================
    params = ParameterSpace(config, wheel, motor, dt)

    params.set_lqr_weights(
        Q=np.diag([10, 1, 1, 1]),
        R=np.array([[0.1]])
    )

    params.set_kalman_covariances(
        W=np.diag([1e-2, 1e-2, 1e-2, 1e-2]),
        V=np.diag([1e-1, 1e-1])
    )

    # ======================================================
    # CONTROLLERS
    # ======================================================
    lqr = LQRController(
        A=params.A,
        B=params.B,
        Q=params.Q,
        R=params.R
    )

    kf = KalmanFilter(
        A=params.A,
        B=params.B,
        C=params.C,
        W=params.W,
        V=params.V
    )

    controller = LQGController(lqr, kf)

    # ======================================================
    # INITIAL STATE
    # ======================================================
    x = np.array([[0.1], [0.0], [0.0], [0.0]])
    u_prev = np.array([[0.0]])

    history = []

    print(
        f"Inizio simulazione | Ip: {plant.I_p:.5f} | Iw: {wheel.inertia:.5f}"
    )

    # ======================================================
    # SIM LOOP
    # ======================================================
    for t in sim.t_steps:

        # measurement
        y = params.C @ x

        # control
        u, x_hat = controller.compute_action(y, u_prev)

        u_prev = u

        history.append(x.flatten())

        # RK4 dynamics
        x_flat = x.flatten()

        k1 = plant.dynamics(x_flat, u)
        k2 = plant.dynamics(x_flat + 0.5 * dt * k1, u)
        k3 = plant.dynamics(x_flat + 0.5 * dt * k2, u)
        k4 = plant.dynamics(x_flat + dt * k3, u)

        x_next = x_flat + (dt / 6.0) * (k1 + 2*k2 + 2*k3 + k4)
        x = x_next.reshape(-1, 1)

    # ======================================================
    # PLOT
    # ======================================================
    history = np.array(history)

    plt.figure(figsize=(12, 6))
    plt.plot(sim.t_steps, history[:, 0], label="Theta")
    plt.plot(sim.t_steps, history[:, 2], label="Phi")
    plt.xlabel("Time [s]")
    plt.ylabel("Angle [rad]")
    plt.title("Closed-loop LQG Control")
    plt.grid(True)
    plt.legend()
    plt.show()


if __name__ == "__main__":
    main()