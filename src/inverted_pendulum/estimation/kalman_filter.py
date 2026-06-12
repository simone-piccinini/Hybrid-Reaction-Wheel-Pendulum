"""Discrete Kalman filter — the LQG state estimator.

The standard predict/update recursion on the discretised plant, feeding the
LQR law ``u = −K x̂`` (the LQG separation, lqr.md "Practical cautions"). Names
follow ``notation.md`` §1/§5: ``W_process`` and ``V_measure`` are the noise
covariances (never the LQR's Q/R), ``x_hat`` the estimate, ``P_est`` the
estimate-error covariance, ``L_gain`` the Kalman gain — distinct from the
control Riccati solution ``P_dare``.

Equations (Anderson & Moore, *Optimal Filtering*, §3.1; Welch & Bishop,
*An Introduction to the Kalman Filter*, eqs. 1.9–1.13):

    predict:  x̂⁻ = A_d x̂ + B_d u
              P⁻  = A_d P A_dᵀ + W
    update:   S   = C P⁻ Cᵀ + V
              L   = P⁻ Cᵀ S⁻¹                        (SPD solve, never inverted)
              x̂   = x̂⁻ + L (z − C x̂⁻)
              P   = (I − L C) P⁻ (I − L C)ᵀ + L V Lᵀ  (Joseph form)

The Joseph-form update preserves symmetry and positive-semidefiniteness under
round-off (numerical_standards.md §4); the ``S⁻¹`` application uses
``chol_solve`` — §3's table reserves that primitive for "GP & Kalman".

Steady state (kalman.md, "Duality note"): the steady-state *a-priori*
covariance solves the dual DARE — ``solve_dare(A_dᵀ, Cᵀ, W, V)`` — so one
Riccati routine serves both the LQR and the estimator. The steady-state gain
exists only for **detectable** ``(A_d, C)`` pairs. ⚠ This plant's locked
sensor set ``[theta_p, theta_w_dot]`` leaves the wheel angle *unobservable*
(column 3 of A is zero and no sensor sees it) at a unit-circle eigenvalue, so
no steady-state filter exists for it: the recursion stays well-defined but
``P_est[2, 2]`` grows without bound — a documented property of the physics,
handled by keeping the horizon finite and the controller's wheel-angle
authority small.

Depends on ``core`` and ``numerics`` only (``dependency_rules.md`` §2).
"""

from __future__ import annotations

from typing import NamedTuple

import numpy as np

from ..core.constants import ATOL
from ..core.types import LQGConfig, StateSpaceModel
from ..numerics.linalg import (
    chol_solve,
    cholesky,
    eigvals,
    is_symmetric,
    symmetrize,
)
from ..numerics.riccati import RICCATI_MAX_ITER, solve_dare


class SteadyStateGain(NamedTuple):
    """Result of :func:`steady_state_kalman_gain`.

    ``L_gain`` is the steady-state filter gain ``P⁻Cᵀ(CP⁻Cᵀ + V)⁻¹``
    (``n_x × n_y``); ``P_pred`` the steady-state a-priori covariance solving
    the dual DARE.
    """

    L_gain: np.ndarray
    P_pred: np.ndarray


def _validate_noise(name: str, M, n: int, *, positive_definite: bool) -> np.ndarray:
    M = np.asarray(M, dtype=np.float64)
    if M.shape != (n, n):
        raise ValueError(f"{name} must have shape ({n}, {n}), got {M.shape}")
    if not is_symmetric(M):
        raise ValueError(f"{name} must be symmetric (within SYM_TOL)")
    if positive_definite:
        cholesky(symmetrize(M))  # raises NotPositiveDefiniteError if not PD
    elif float(np.min(eigvals(symmetrize(M)).real)) < -ATOL:
        raise ValueError(f"{name} must be positive semidefinite")
    return symmetrize(M)


def steady_state_kalman_gain(
    model: StateSpaceModel, W_process, V_measure, *, max_iter: int = RICCATI_MAX_ITER
) -> SteadyStateGain:
    """Steady-state Kalman gain via the dual DARE (kalman.md, "Duality note").

    Substitutes ``A → A_dᵀ, B → Cᵀ, Q → W, R → V`` into the one Riccati
    routine: the converged ``P_pred`` satisfies

        P⁻ = W + A P⁻ Aᵀ − A P⁻ Cᵀ (C P⁻ Cᵀ + V)⁻¹ C P⁻ Aᵀ,

    and the gain is read off it. Requires a **detectable** ``(A_d, C)`` —
    see the module docstring for this plant's caveat.

    Raises
    ------
    ValueError
        If the model is continuous or the covariances are malformed.
    RiccatiNotConverged
        Propagated from ``solve_dare`` (e.g. an undetectable pair).
    """
    if not isinstance(model, StateSpaceModel):
        raise TypeError("model must be a StateSpaceModel")
    if not model.is_discrete:
        raise ValueError(
            "the Kalman recursions run in discrete time: pass the discretised "
            "model (model.discretize(dt)), not the continuous one"
        )
    W = _validate_noise("W_process", W_process, model.n_x, positive_definite=False)
    V = _validate_noise("V_measure", V_measure, model.n_y, positive_definite=True)
    solution = solve_dare(model.A.T, model.C.T, W, V, max_iter=max_iter)
    P_pred = solution.P
    S = symmetrize(V + model.C @ P_pred @ model.C.T)
    L_gain = chol_solve(S, model.C @ P_pred).T  # (n_x, n_y)
    return SteadyStateGain(L_gain=L_gain, P_pred=P_pred)


class KalmanFilter:
    """The recursive discrete Kalman filter (``class_diagram.md``; ``notation.md`` §5).

    Mutable by design — ``x_hat`` and ``P_est`` evolve with each
    :meth:`predict`/:meth:`update` — but fully deterministic: the trajectory
    of estimates is a pure function of ``(x0, P0, u-sequence, z-sequence)``.

    Parameters
    ----------
    model : StateSpaceModel
        The **discrete** linearised plant (provides ``A_d``, ``B_d``, ``C``).
    W_process : (n_x, n_x) array_like
        Process-noise covariance ``W``, symmetric PSD.
    V_measure : (n_y, n_y) array_like
        Measurement-noise covariance ``V``, symmetric PD.
    x0 : (n_x,) array_like, optional
        Initial estimate (default zeros — the upright equilibrium).
    P0 : (n_x, n_x) array_like, optional
        Initial covariance (default ``W``, the one-step prior uncertainty).
    """

    def __init__(self, model, W_process, V_measure, x0=None, P0=None) -> None:
        if not isinstance(model, StateSpaceModel):
            raise TypeError("model must be a StateSpaceModel")
        if not model.is_discrete:
            raise ValueError(
                "the Kalman recursions run in discrete time: pass the "
                "discretised model (model.discretize(dt)), not the continuous one"
            )
        self.model = model
        self.W_process = _validate_noise(
            "W_process", W_process, model.n_x, positive_definite=False
        )
        self.V_measure = _validate_noise(
            "V_measure", V_measure, model.n_y, positive_definite=True
        )
        self.W_process.flags.writeable = False
        self.V_measure.flags.writeable = False

        x0 = np.zeros(model.n_x) if x0 is None else np.asarray(x0, dtype=np.float64)
        if x0.shape != (model.n_x,):
            raise ValueError(f"x0 must have shape ({model.n_x},), got {x0.shape}")
        P0 = self.W_process if P0 is None else P0
        self.x_hat = x0.copy()
        self.P_est = _validate_noise("P0", P0, model.n_x, positive_definite=False)

    @classmethod
    def from_config(
        cls, model: StateSpaceModel, config: LQGConfig, x0=None, P0=None
    ) -> "KalmanFilter":
        """Build from an :class:`LQGConfig` — the typed hand-off.

        Consumes ``config.W_process`` and ``config.V_measure``; the LQR
        weights in the same config belong to the controller (``notation.md`` §1).
        """
        if not isinstance(config, LQGConfig):
            raise TypeError("config must be an LQGConfig")
        return cls(model, config.W_process, config.V_measure, x0=x0, P0=P0)

    @property
    def n_x(self) -> int:
        """State dimension."""
        return self.model.n_x

    @property
    def n_y(self) -> int:
        """Measurement dimension."""
        return self.model.n_y

    @property
    def state_estimate(self) -> np.ndarray:
        """A copy of the current estimate ``x̂`` (``class_diagram.md``)."""
        return self.x_hat.copy()

    def predict(self, u) -> np.ndarray:
        """Time update: propagate the estimate through the model dynamics.

        ``x̂⁻ = A_d x̂ + B_d u``, ``P⁻ = A_d P A_dᵀ + W`` (Welch & Bishop
        eqs. 1.9–1.10). Returns a copy of the predicted estimate.
        """
        u = np.atleast_1d(np.asarray(u, dtype=np.float64))
        if u.shape != (self.model.n_u,):
            raise ValueError(f"u must have shape ({self.model.n_u},), got {u.shape}")
        A, B = self.model.A, self.model.B
        self.x_hat = A @ self.x_hat + B @ u
        self.P_est = symmetrize(A @ self.P_est @ A.T + self.W_process)
        return self.x_hat.copy()

    def update(self, z) -> np.ndarray:
        """Measurement update: correct the estimate with the innovation.

        ``L = P⁻Cᵀ(CP⁻Cᵀ + V)⁻¹`` applied as an SPD solve (never an explicit
        inverse), ``x̂ += L(z − Cx̂⁻)``, Joseph-form covariance (Welch & Bishop
        eqs. 1.11–1.13; Anderson & Moore §3.1). Returns a copy of the
        corrected estimate.
        """
        z = np.asarray(z, dtype=np.float64)
        if z.shape != (self.model.n_y,):
            raise ValueError(f"z must have shape ({self.model.n_y},), got {z.shape}")
        C, V = self.model.C, self.V_measure
        P_prior = self.P_est
        S = symmetrize(C @ P_prior @ C.T + V)
        L_gain = chol_solve(S, C @ P_prior).T  # (n_x, n_y)
        innovation = z - C @ self.x_hat
        self.x_hat = self.x_hat + L_gain @ innovation
        closed = np.eye(self.n_x) - L_gain @ C
        self.P_est = symmetrize(
            closed @ P_prior @ closed.T + L_gain @ V @ L_gain.T
        )
        return self.x_hat.copy()

    def step(self, u, z) -> np.ndarray:
        """One full LQG cycle: :meth:`predict` with ``u``, then :meth:`update` with ``z``."""
        self.predict(u)
        return self.update(z)
