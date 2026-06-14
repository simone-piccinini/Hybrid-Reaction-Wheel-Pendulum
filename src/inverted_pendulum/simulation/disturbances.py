"""True-world disturbances — the noise the simulation actually injects.

These are properties of the simulated *environment*, deliberately distinct
from the ``W_process``/``V_measure`` inside :class:`LQGConfig`: the config
carries what the Kalman filter **assumes**, a tuning parameter the Bayesian
optimiser searches; the disturbances here are what the world **does**. Were
the true noise taken from the candidate config, the optimiser could cheat by
proposing a noiseless world.

All draws flow from the caller's seeded ``numpy.random.Generator``
(AGENTS §7 / numerical_standards.md §6 — no module-level randomness, no
legacy ``np.random.*``). Noise is modelled with diagonal covariances —
per-component standard deviations — matching the diagonal-only convention of
the LQG weight packing (``data_contracts.md`` §1).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


def _as_std_vector(name: str, values, n: int | None) -> np.ndarray:
    arr = np.atleast_1d(np.asarray(values, dtype=np.float64))
    if arr.ndim != 1:
        raise ValueError(f"{name} must be a 1-D vector, got shape {arr.shape}")
    if n is not None and arr.shape[0] != n:
        raise ValueError(f"{name} must have length {n}, got {arr.shape[0]}")
    if np.any(arr < 0.0):
        raise ValueError(f"{name} entries are standard deviations, must be >= 0")
    arr.flags.writeable = False
    return arr


@dataclass(frozen=True)
class Disturbances:
    """Per-component Gaussian noise levels for one simulated world.

    Parameters
    ----------
    process_std : (n_x,) array_like
        Standard deviation of the additive process noise applied to the true
        state once per step, ≥ 0 elementwise (zeros disable a channel).
    measurement_std : (n_y,) array_like
        Standard deviation of the additive sensor noise, ≥ 0 elementwise.
    initial_state_std : (n_x,) array_like, optional
        Standard deviation of the initial-state perturbation drawn once per
        rollout around the nominal initial state (AGENTS §7 lists it among
        the seeded randomness). Defaults to zeros (deterministic start).
    """

    process_std: np.ndarray
    measurement_std: np.ndarray
    initial_state_std: np.ndarray | None = field(default=None)

    def __post_init__(self) -> None:
        process = _as_std_vector("process_std", self.process_std, None)
        measurement = _as_std_vector("measurement_std", self.measurement_std, None)
        if self.initial_state_std is None:
            initial = np.zeros(process.shape[0])
            initial.flags.writeable = False
        else:
            initial = _as_std_vector(
                "initial_state_std", self.initial_state_std, process.shape[0]
            )
        object.__setattr__(self, "process_std", process)
        object.__setattr__(self, "measurement_std", measurement)
        object.__setattr__(self, "initial_state_std", initial)

    @property
    def n_x(self) -> int:
        """State dimension the process noise applies to."""
        return self.process_std.shape[0]

    @property
    def n_y(self) -> int:
        """Measurement dimension the sensor noise applies to."""
        return self.measurement_std.shape[0]

    def process_noise(self, rng: np.random.Generator) -> np.ndarray:
        """One draw ``w ~ N(0, diag(process_std²))`` from the threaded generator."""
        return self.process_std * rng.standard_normal(self.n_x)

    def measurement_noise(self, rng: np.random.Generator) -> np.ndarray:
        """One draw ``v ~ N(0, diag(measurement_std²))`` from the threaded generator."""
        return self.measurement_std * rng.standard_normal(self.n_y)

    def initial_perturbation(self, rng: np.random.Generator) -> np.ndarray:
        """One draw of the initial-state perturbation (AGENTS §7)."""
        return self.initial_state_std * rng.standard_normal(self.n_x)
