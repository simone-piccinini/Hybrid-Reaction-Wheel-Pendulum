"""Shared fixtures for the unit suite."""

import numpy as np
import pytest

from inverted_pendulum.core.types import LQGConfig, SimulationResult


@pytest.fixture
def make_result():
    """Factory for synthetic SimulationResults with the project's (4,1,2) dims.

    Pass ``theta`` (1-D) to plant a trajectory in state column 0, or full
    ``states`` (T,4); other arrays default to zeros on a uniform ``dt`` grid.
    """

    def factory(theta=None, *, states=None, controls=None, time=None,
                dt=0.01, diverged=False):
        if states is None:
            theta_arr = np.asarray(theta, dtype=np.float64)
            horizon = theta_arr.shape[0]
            states = np.zeros((horizon, 4))
            states[:, 0] = theta_arr
        else:
            states = np.asarray(states, dtype=np.float64)
            horizon = states.shape[0]
        if controls is None:
            controls = np.zeros((horizon, 1))
        if time is None:
            time = dt * np.arange(horizon)
        config = LQGConfig(
            Q_lqr=np.eye(4), R_lqr=np.eye(1),
            W_process=1e-3 * np.eye(4), V_measure=1e-2 * np.eye(2),
        )
        return SimulationResult(
            time=time, true_states=states, estimated_states=states,
            controls=controls, measurements=np.zeros((horizon, 2)),
            seed=0, diverged=diverged, config=config,
        )

    return factory
