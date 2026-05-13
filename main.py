import numpy as np
import matplotlib.pyplot as plt

from core.config_loader import PhysicalConfig
from core.components import ReactionWheel, DCMotor
from core.plant import ReactionWheelPendulum
from core.simulation_engine import SimulationEngine


def main():

    config = PhysicalConfig.from_yaml("config/params.yaml")

    my_wheel = ReactionWheel(
        mass=config.wheel.m,
        radius=config.wheel.r,
        friction=config.wheel.b
    )

    my_motor = DCMotor(
        K_t=config.motor.K_t,
        K_e=config.motor.K_e,
        R_a=config.motor.R_a
    )

    pendulum = ReactionWheelPendulum(config, my_wheel, my_motor)

    sim = SimulationEngine(
        plant=pendulum,
        dt=0.01,
        t_end=5.0
    )

    x0 = np.array([0.1, 0.0, 0.0, 0.0])

    print(
        f"Inizio simulazione con "
        f"Ip: {pendulum.I_p:.5f} "
        f"e Iw: {my_wheel.inertia:.5f}"
    )

    # zero voltage input
    def input_function(t, x):
        return 0.0

    t_steps, history = sim.run(x0, input_function)

    # visualization
    plt.figure(figsize=(10, 6))

    plt.plot(t_steps, history[:, 0],
             label='Theta (Angolo Pendolo)')

    plt.plot(t_steps, history[:, 2],
             label='Phi (Angolo Ruota)')

    plt.xlabel('Tempo [s]')
    plt.ylabel('Angolo [rad]')
    plt.title('Simulazione Pendolo Non Controllato')
    plt.legend()
    plt.grid(True)

    plt.show()


if __name__ == "__main__":
    main()