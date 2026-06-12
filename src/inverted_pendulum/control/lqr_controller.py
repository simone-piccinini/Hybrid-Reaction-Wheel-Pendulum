"""LQR state-feedback controller — gain design via the discrete Riccati equation.

Implements ``docs/theory/lqr.md``: the constant gain ``K`` is read off the
stabilising DARE solution computed by :func:`numerics.riccati.solve_dare`
(value iteration, ``numerical_standards.md`` §5), and the control law applied
every step is the single matrix multiply

    u = −K x̂                                    (lqr.md, "Control law")

on the **state estimate** — the LQG separation: LQR consumes ``x̂`` from the
Kalman filter, not the true state. ``Q_lqr`` and ``R_lqr`` are tuning
parameters owned by the optimization layer (``notation.md`` §1, §4), not
physical constants; this module treats them as inputs.

Per lqr.md's "Practical cautions" the design confirms the closed loop is
stable: all eigenvalues of ``A_d − B_d K`` strictly inside the unit circle,
checked with the ``numerics`` shifted-QR eigensolver. A failure raises
:class:`UnstableClosedLoopError` — a *configuration* error the optimizer maps
to a large finite penalty rather than crashing (``numerical_standards.md`` §7).

Depends on ``core`` and ``numerics`` only (``dependency_rules.md`` §2).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..core.constants import ATOL
from ..core.types import LQGConfig, StateSpaceModel
from ..numerics.linalg import spectral_radius
from ..numerics.riccati import RICCATI_MAX_ITER, solve_dare


class UnstableClosedLoopError(Exception):
    """The designed gain leaves an eigenvalue of ``A_d − B_d K`` on/outside the unit circle.

    lqr.md, "Practical cautions": treat as a configuration error; the
    optimization layer maps it to a large finite penalty.
    """


@dataclass(frozen=True)
class LQRController:
    """The designed LQR law (``class_diagram.md``; ``notation.md`` §4).

    Build with :meth:`from_model` (explicit weights) or :meth:`from_config`
    (an :class:`LQGConfig`, the typed hand-off of ``data_contracts.md``).
    Fields are read-only float64; the dataclass is immutable so a designed
    controller cannot drift mid-simulation (determinism contract).

    Fields
    ------
    Q_lqr, R_lqr : ndarray
        The state and control cost weights the gain was designed from.
    K_gain : ndarray
        Optimal feedback gain ``K`` (``n_u × n_x``).
    P_dare : ndarray
        Converged Riccati solution ``P`` (``n_x × n_x``) — ``notation.md``
        §4's control Riccati solution, suffixed ``_dare`` for the discrete
        formulation this project locked in (distinct from the estimator's
        ``P_est``, never bare ``P``).
    closed_loop_spectral_radius : float
        ``ρ(A_d − B_d K)`` recorded at design time (< 1 by construction).
    """

    Q_lqr: np.ndarray
    R_lqr: np.ndarray
    K_gain: np.ndarray
    P_dare: np.ndarray
    closed_loop_spectral_radius: float

    def __post_init__(self) -> None:
        for name in ("Q_lqr", "R_lqr", "K_gain", "P_dare"):
            arr = np.array(getattr(self, name), dtype=np.float64)  # copy -> read-only
            arr.flags.writeable = False
            object.__setattr__(self, name, arr)
        object.__setattr__(
            self,
            "closed_loop_spectral_radius",
            float(self.closed_loop_spectral_radius),
        )
        n_x = self.Q_lqr.shape[0]
        n_u = self.R_lqr.shape[0]
        if self.Q_lqr.shape != (n_x, n_x) or self.R_lqr.shape != (n_u, n_u):
            raise ValueError("Q_lqr and R_lqr must be square")
        if self.K_gain.shape != (n_u, n_x):
            raise ValueError(
                f"K_gain must be (n_u, n_x)=({n_u}, {n_x}), got {self.K_gain.shape}"
            )
        if self.P_dare.shape != (n_x, n_x):
            raise ValueError(
                f"P_dare must be (n_x, n_x)=({n_x}, {n_x}), got {self.P_dare.shape}"
            )
        # lqr.md caution: an eigenvalue on the unit circle does not stabilise;
        # "on" is judged within the §2 absolute tolerance.
        if self.closed_loop_spectral_radius >= 1.0 - ATOL:
            raise UnstableClosedLoopError(
                f"closed-loop spectral radius {self.closed_loop_spectral_radius!r} "
                "is not strictly inside the unit circle"
            )

    @classmethod
    def from_model(
        cls, model: StateSpaceModel, Q_lqr, R_lqr, *, max_iter: int = RICCATI_MAX_ITER
    ) -> "LQRController":
        """Design the gain for a **discrete** plant model (lqr.md, "Implementation order").

        Solves the DARE on ``(A_d, B_d)`` with the given weights and verifies
        the closed loop. ``solve_dare`` validates the weights (``Q`` symmetric
        PSD, ``R`` symmetric PD) and shapes; ``max_iter`` is forwarded to the
        value iteration (weakly-controlled modes converge slowly).

        Raises
        ------
        ValueError
            If the model is continuous (lqr.md consumes the *discretised*
            matrices — ``data_contracts.md`` §6: check ``is_discrete``).
        UnstableClosedLoopError
            If the resulting closed loop is not strictly stable.
        RiccatiNotConverged
            Propagated from ``solve_dare``.
        """
        if not isinstance(model, StateSpaceModel):
            raise TypeError("model must be a StateSpaceModel")
        if not model.is_discrete:
            raise ValueError(
                "LQR runs in discrete time: pass the discretised model "
                "(model.discretize(dt)), not the continuous one (lqr.md)"
            )
        solution = solve_dare(model.A, model.B, Q_lqr, R_lqr, max_iter=max_iter)
        rho = spectral_radius(model.A - model.B @ solution.K)
        return cls(
            Q_lqr=np.asarray(Q_lqr, dtype=np.float64),
            R_lqr=np.asarray(R_lqr, dtype=np.float64),
            K_gain=solution.K,
            P_dare=solution.P,
            closed_loop_spectral_radius=rho,
        )

    @classmethod
    def from_config(
        cls,
        model: StateSpaceModel,
        config: LQGConfig,
        *,
        max_iter: int = RICCATI_MAX_ITER,
    ) -> "LQRController":
        """Design from an :class:`LQGConfig` — the typed boundary object.

        Consumes ``config.Q_lqr`` and ``config.R_lqr``; the Kalman weights
        ``(W_process, V_measure)`` in the same config belong to the estimator
        (``notation.md`` §1) and are ignored here.
        """
        if not isinstance(config, LQGConfig):
            raise TypeError("config must be an LQGConfig")
        return cls.from_model(model, config.Q_lqr, config.R_lqr, max_iter=max_iter)

    @property
    def n_x(self) -> int:
        """State dimension."""
        return self.K_gain.shape[1]

    @property
    def n_u(self) -> int:
        """Input dimension."""
        return self.K_gain.shape[0]

    def compute_control(self, x_hat) -> np.ndarray:
        """The control law ``u = −K x̂`` (lqr.md; ``notation.md`` §4).

        ``x_hat`` is the Kalman state estimate (or the true state in
        full-state-feedback tests). Returns ``u`` with shape ``(n_u,)``.
        """
        x = np.asarray(x_hat, dtype=np.float64)
        if x.shape != (self.n_x,):
            raise ValueError(f"x_hat must have shape ({self.n_x},), got {x.shape}")
        return -(self.K_gain @ x)
