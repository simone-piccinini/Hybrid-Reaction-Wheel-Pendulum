"""Linearise-and-discretise pipeline — the design model for LQR and Kalman.

Implements steps 5–6 of ``model.md``'s "Implementation order": take the
continuous upright linearisation produced by
:meth:`ReactionWheelPendulum.linearize`, discretise it at the control sampling
period via the matrix-exponential primitive, and hand off ``(A_d, B_d, C)`` to
the control and estimation layers. Keeps the continuous and discrete models
**distinct and together**, per ``model.md``'s caution: "Do not mix them."

Exposes the same ``step(state, voltage)`` protocol as
:class:`~inverted_pendulum.dynamics.nonlinear_model.NonlinearPlantModel`, so
the simulation layer can swap the true plant for its linear approximation —
under zero-order-hold the discrete update ``x⁺ = A_d x + B_d u`` is **exact**
for the linear dynamics (no integrator truncation error).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..core.types import StateSpaceModel
from ..physical.pendulum import ReactionWheelPendulum


@dataclass(frozen=True)
class LinearizedPlantModel:
    """The continuous linearisation and its ZOH discretisation, as a pair.

    Build with :meth:`from_plant`; the two-model constructor exists for tests
    and deserialisation. Invariants checked at construction: ``continuous`` is
    a continuous-time model, ``discrete`` a discrete-time one, with identical
    dimensions and identical output map ``(C, D)`` (discretisation carries the
    measurement model over unchanged — ``data_contracts.md`` §6).
    """

    continuous: StateSpaceModel
    discrete: StateSpaceModel

    def __post_init__(self) -> None:
        if not isinstance(self.continuous, StateSpaceModel) or not isinstance(
            self.discrete, StateSpaceModel
        ):
            raise TypeError("continuous and discrete must be StateSpaceModel")
        if self.continuous.is_discrete:
            raise ValueError("continuous model must have is_discrete=False")
        if not self.discrete.is_discrete:
            raise ValueError("discrete model must have is_discrete=True")
        same_dims = (
            self.continuous.n_x == self.discrete.n_x
            and self.continuous.n_u == self.discrete.n_u
            and self.continuous.n_y == self.discrete.n_y
        )
        if not same_dims:
            raise ValueError("continuous and discrete dimensions differ")
        if not (
            np.array_equal(self.continuous.C, self.discrete.C)
            and np.array_equal(self.continuous.D, self.discrete.D)
        ):
            raise ValueError(
                "continuous and discrete must share the output map (C, D)"
            )

    @classmethod
    def from_plant(
        cls, plant: ReactionWheelPendulum, dt: float, operating_point=None
    ) -> "LinearizedPlantModel":
        """``model.md`` steps 3–6: assemble ``(A, B, C)``, then discretise at ``dt``.

        ``operating_point`` is forwarded to
        :meth:`ReactionWheelPendulum.linearize` (only the upright equilibrium
        is supported). ``dt`` is the control sampling period from config.
        """
        if not isinstance(plant, ReactionWheelPendulum):
            raise TypeError("plant must be a ReactionWheelPendulum")
        continuous = plant.linearize(operating_point)
        return cls(continuous=continuous, discrete=continuous.discretize(dt))

    # ------------------------------------------------------------------ #
    # hand-off accessors (model.md step 6; notation.md §3 names)
    # ------------------------------------------------------------------ #
    @property
    def A_d(self) -> np.ndarray:
        """Discrete state matrix ``A_d``."""
        return self.discrete.A

    @property
    def B_d(self) -> np.ndarray:
        """Discrete input matrix ``B_d``."""
        return self.discrete.B

    @property
    def C(self) -> np.ndarray:
        """Output matrix ``C`` (identical for both models)."""
        return self.discrete.C

    @property
    def dt(self) -> float:
        """The control sampling period the discrete model was built at."""
        return self.discrete.dt

    @property
    def n_x(self) -> int:
        """State dimension."""
        return self.discrete.n_x

    @property
    def n_u(self) -> int:
        """Input dimension."""
        return self.discrete.n_u

    @property
    def n_y(self) -> int:
        """Output dimension."""
        return self.discrete.n_y

    def step(self, state, voltage: float) -> np.ndarray:
        """One exact ZOH step of the linear dynamics: ``x⁺ = A_d x + B_d u``.

        Same protocol as :meth:`NonlinearPlantModel.step`; ``voltage`` is the
        scalar input of this single-input plant (``model.md``). Note the
        linear model knows nothing of the motor's saturations — those exist
        only in the nonlinear plant.
        """
        x = np.asarray(state, dtype=np.float64)
        if x.shape != (self.n_x,):
            raise ValueError(f"state must have shape ({self.n_x},), got {x.shape}")
        return self.A_d @ x + self.B_d[:, 0] * float(voltage)
