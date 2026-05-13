# core/simulation_engine.py

import numpy as np


class SimulationEngine:
    def __init__(self, plant, dt: float, t_end: float):
        """
        Generic simulation engine using RK4 integration.

        Parameters
        ----------
        plant : object
            Must expose a method:
                dynamics(x, u) -> dx/dt

        dt : float
            Simulation timestep.

        t_end : float
            Total simulation duration.
        """
        self.plant = plant
        self.dt = dt
        self.t_end = t_end
        self.t_steps = np.arange(0, t_end, dt)

    def rk4_step(self, x, u):
        """
        Perform a single RK4 integration step.
        """
        k1 = self.plant.dynamics(x, u)
        k2 = self.plant.dynamics(x + 0.5 * self.dt * k1, u)
        k3 = self.plant.dynamics(x + 0.5 * self.dt * k2, u)
        k4 = self.plant.dynamics(x + self.dt * k3, u)

        return x + (self.dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)

    def run(self, x0, input_function=None):
        """
        Run the simulation.

        Parameters
        ----------
        x0 : np.ndarray
            Initial state vector.

        input_function : callable
            Function of the form:
                u = input_function(t, x)

            If None, zero input is used.

        Returns
        -------
        t_steps : np.ndarray
            Simulation time vector.

        history : np.ndarray
            State history over time.
        """

        x = np.array(x0, dtype=float)
        history = []

        for t in self.t_steps:

            if input_function is None:
                u = 0.0
            else:
                u = input_function(t, x)

            history.append(x.copy())

            x = self.rk4_step(x, u)

        return self.t_steps, np.array(history)