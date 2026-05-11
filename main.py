import numpy as np
import matplotlib.pyplot as plt
from core.config_loader import PhysicalConfig
from core.components import ReactionWheel, DCMotor # Assicurati che DCMotor sia in components.py
from core.plant import ReactionWheelPendulum

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

    # --- SIMULATION ---
    dt = 0.01          # Step temporale (10ms, tipico per STM32)
    t_end = 5.0        # Simula per 5 secondi
    t_steps = np.arange(0, t_end, dt)
    
    # Stato iniziale: [theta, th_dot, phi, ph_dot]
    # Iniziamo con il pendolo leggermente fuori asse (0.1 rad)
    x = np.array([0.1, 0.0, 0.0, 0.0])
    
    history = []

    print(f"Inizio simulazione con Ip: {pendulum.I_p:.5f} e Iw: {my_wheel.inertia:.5f}")

    for t in t_steps:
        # zero tension
        V_a = 0.0 
        
        # save actual state
        history.append(x.copy())
        
        # RK4
        k1 = pendulum.dynamics(x, V_a)
        k2 = pendulum.dynamics(x + 0.5 * dt * k1, V_a)
        k3 = pendulum.dynamics(x + 0.5 * dt * k2, V_a)
        k4 = pendulum.dynamics(x + dt * k3, V_a)
        
        x = x + (dt / 6.0) * (k1 + 2*k2 + 2*k3 + k4)

    # visualizing
    history = np.array(history)
    plt.figure(figsize=(10, 6))
    plt.plot(t_steps, history[:, 0], label='Theta (Angolo Pendolo)')
    plt.plot(t_steps, history[:, 2], label='Phi (Angolo Ruota)')
    plt.xlabel('Tempo [s]')
    plt.ylabel('Angolo [rad]')
    plt.title('Simulazione Pendolo Non Controllato (Caduta Libera)')
    plt.legend()
    plt.grid(True)
    plt.show()

if __name__ == "__main__":
    main()