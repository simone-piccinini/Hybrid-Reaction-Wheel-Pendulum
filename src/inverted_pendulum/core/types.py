"""Domain value objects — the typed contracts passed between layers.

Implemented as frozen dataclasses with read-only float64 arrays, supporting the
determinism contract (``data_contracts.md``; ``AGENTS.md`` §7). ``core`` may
import ``numerics`` (``dependency_rules.md`` §2) — :meth:`StateSpaceModel.discretize`
uses the matrix-exponential primitive.

This module currently provides :class:`StateSpaceModel`. The remaining contracts
(``LQGConfig``, ``SearchSpace``, ``SimulationResult``, ``Dataset``,
``GPPosterior``) and the abstract interfaces will be added in subsequent files.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..numerics.matrix_exp import expm


@dataclass(frozen=True)
class StateSpaceModel:
    """Linear time-invariant state-space model (``data_contracts.md`` §6).

    Continuous: ``ẋ = A x + B u``, ``y = C x + D u``. Discrete (one step):
    ``x⁺ = A x + B u``, ``y = C x + D u`` when ``is_discrete``. All arrays are
    coerced to read-only float64 (immutability supports determinism). Shapes use
    the ``notation.md`` symbols: ``A`` is ``n_x × n_x``, ``B`` is ``n_x × n_u``,
    ``C`` is ``n_y × n_x``, ``D`` is ``n_y × n_u``.
    """

    A: np.ndarray
    B: np.ndarray
    C: np.ndarray
    D: np.ndarray
    is_discrete: bool = False
    dt: float | None = None

    def __post_init__(self) -> None:
        A = np.array(self.A, dtype=np.float64)  # copy: we make these read-only
        B = np.array(self.B, dtype=np.float64)
        C = np.array(self.C, dtype=np.float64)
        D = np.array(self.D, dtype=np.float64)
        if A.ndim != 2 or A.shape[0] != A.shape[1]:
            raise ValueError(f"A must be square (n_x, n_x), got {A.shape}")
        n_x = A.shape[0]
        if B.ndim != 2 or B.shape[0] != n_x:
            raise ValueError(f"B must be (n_x, n_u) with n_x={n_x}, got {B.shape}")
        n_u = B.shape[1]
        if C.ndim != 2 or C.shape[1] != n_x:
            raise ValueError(f"C must be (n_y, n_x) with n_x={n_x}, got {C.shape}")
        n_y = C.shape[0]
        if D.shape != (n_y, n_u):
            raise ValueError(f"D must be (n_y, n_u)=({n_y}, {n_u}), got {D.shape}")
        if self.is_discrete:
            if self.dt is None or self.dt <= 0:
                raise ValueError("a discrete model requires dt > 0")
        elif self.dt is not None:
            raise ValueError("a continuous model must not set dt")
        for arr in (A, B, C, D):
            arr.flags.writeable = False
        object.__setattr__(self, "A", A)
        object.__setattr__(self, "B", B)
        object.__setattr__(self, "C", C)
        object.__setattr__(self, "D", D)

    @property
    def n_x(self) -> int:
        """State dimension."""
        return self.A.shape[0]

    @property
    def n_u(self) -> int:
        """Input dimension."""
        return self.B.shape[1]

    @property
    def n_y(self) -> int:
        """Output dimension."""
        return self.C.shape[0]

    def discretize(self, dt: float) -> "StateSpaceModel":
        """Return a **new** zero-order-hold discrete model at step ``dt``.

        Uses the augmented matrix exponential (``model.md``, "Discretise";
        ``data_contracts.md`` §6):

            expm([[A, B], [0, 0]] · dt) = [[A_d, B_d], [0, I]],

        so ``A_d`` and ``B_d`` are the top-left and top-right blocks. ``C`` and
        ``D`` carry over unchanged. Does not mutate ``self``.

        Raises
        ------
        ValueError
            If the model is already discrete, or ``dt <= 0``.
        """
        if self.is_discrete:
            raise ValueError("model is already discrete")
        if dt <= 0:
            raise ValueError("dt must be > 0")
        n_x, n_u = self.n_x, self.n_u
        aug = np.zeros((n_x + n_u, n_x + n_u), dtype=np.float64)
        aug[:n_x, :n_x] = self.A
        aug[:n_x, n_x:] = self.B
        transition = expm(aug * dt)
        A_d = transition[:n_x, :n_x]
        B_d = transition[:n_x, n_x:]
        return StateSpaceModel(A_d, B_d, self.C, self.D, is_discrete=True, dt=dt)
