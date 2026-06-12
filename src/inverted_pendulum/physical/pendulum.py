"""Reaction-wheel pendulum plant — nonlinear dynamics and upright linearisation.

The plant of ``docs/theory/model.md``: a pendulum free to swing with a
motor-driven reaction wheel mounted on it. The motor spins the wheel and the
wheel's reaction torque on the body is the only control authority. State
ordering follows ``notation.md`` §3:

    x = [theta_p, theta_p_dot, theta_w, theta_w_dot],   u = voltage.

The nonlinear equations of motion (the model whose ``θ = 0`` linearisation is
exactly ``model.md``'s explicit ``A``, ``B``):

    I_p · θ̈           = m_p g ℓ_p sin θ − b_p θ̇ − τ_w        (pendulum body)
    I_w · (θ̈ + φ̈)     = τ_w                                   (wheel, absolute)
    τ_w               = τ_m − b_w φ̇                            (net wheel torque)

where ``τ_m`` is the electromagnetic motor torque and ``θ̈ + φ̈`` is the wheel's
absolute angular acceleration (``φ`` is measured relative to the pendulum).
Both ``τ_m`` and the bearing friction are internal pendulum–wheel torques, so
``−τ_w`` is the reaction on the body — the action–reaction coupling that gives
``model.md``'s opposite signs between rows 2 and 4. The wheel angle ``theta_w``
does not enter the dynamics (column 3 of ``A`` is zero); it is uncontrollable
and retained only for tracking.

Symbols ↔ code names follow ``notation.md`` §2–§3 (``model.md``'s ``I_p`` is
``body_inertia``, its ``m``/``l`` are ``pendulum_mass``/``pendulum_length``).
Depends on ``core``, ``numerics`` (via ``core``), and numpy only
(``dependency_rules.md`` §2).
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from ..core.constants import ATOL, GRAVITY
from ..core.types import StateSpaceModel
from .motor import DCMotor
from .wheel import ReactionWheel

N_X: int = 4
"""State dimension of the plant (``notation.md`` §3)."""


@dataclass(frozen=True)
class ReactionWheelPendulum:
    """The composed plant: body + :class:`ReactionWheel` + :class:`DCMotor`.

    Parameters (``notation.md`` §2; ``model.md`` "Parameters the model needs")
    ----------
    pendulum_mass : float
        System mass ``m_p`` (kg), > 0 (``model.md``'s ``m``).
    pendulum_length : float
        Pivot→centre-of-mass distance ``ℓ_p`` (m), > 0 (``model.md``'s ``l``).
    body_inertia : float
        Pendulum moment of inertia ``I_b`` about the pivot (kg·m²), > 0 — it is
        a denominator (``model.md``'s ``I_p``).
    wheel : ReactionWheel
        The reaction wheel (supplies ``I_w`` and ``b_w``).
    motor : DCMotor
        The actuator (supplies ``E = K_t/R_a`` and the back-EMF damping).
        Its ``friction_coefficient`` is **not** part of ``model.md``'s model;
        lump any motor bearing friction into the wheel's ``b_w``.
    pivot_friction : float
        Pendulum pivot viscous-friction coefficient ``b_p`` (N·m·s/rad), ≥ 0.
    gravity : float
        Gravitational acceleration ``g`` (m/s²), > 0.
    """

    pendulum_mass: float
    pendulum_length: float
    body_inertia: float
    wheel: ReactionWheel
    motor: DCMotor
    pivot_friction: float = 0.0
    gravity: float = GRAVITY

    def __post_init__(self) -> None:
        for name in ("pendulum_mass", "pendulum_length", "body_inertia",
                     "pivot_friction", "gravity"):
            object.__setattr__(self, name, float(getattr(self, name)))
        if self.pendulum_mass <= 0.0:
            raise ValueError("pendulum_mass m_p must be > 0")
        if self.pendulum_length <= 0.0:
            raise ValueError("pendulum_length ℓ_p must be > 0")
        if self.body_inertia <= 0.0:
            raise ValueError("body_inertia I_b must be > 0 (it is a denominator)")
        if self.pivot_friction < 0.0:
            raise ValueError("pivot_friction b_p must be >= 0")
        if self.gravity <= 0.0:
            raise ValueError("gravity g must be > 0")
        if not isinstance(self.wheel, ReactionWheel):
            raise TypeError("wheel must be a ReactionWheel")
        if not isinstance(self.motor, DCMotor):
            raise TypeError("motor must be a DCMotor")

    # ------------------------------------------------------------------ #
    # lumped scalars (model.md, "Parameters the model needs")
    # ------------------------------------------------------------------ #
    @property
    def wheel_inertia(self) -> float:
        """``I_w`` — the wheel's moment of inertia (``notation.md`` §2)."""
        return self.wheel.inertia

    @property
    def voltage_to_torque_gain(self) -> float:
        """``E = K_t / R_a`` — voltage→torque gain (``model.md``)."""
        return self.motor.voltage_to_torque_gain

    @property
    def lumped_damping(self) -> float:
        """``D = K_t K_e / R_a + b_w`` — back-EMF damping + wheel friction (``model.md``)."""
        return self.motor.back_emf_damping + self.wheel.friction_coefficient

    # ------------------------------------------------------------------ #
    # dynamics
    # ------------------------------------------------------------------ #
    def nonlinear_dynamics(self, state, input_torque: float) -> np.ndarray:
        """Full nonlinear ``ẋ = f(x, τ_m)`` with the motor torque as input.

        Implements the equations of motion in the module docstring
        (``model.md``, gravity entering as ``sin θ``). ``input_torque`` is the
        electromagnetic motor torque ``τ_m`` on the wheel; bearing friction is
        applied internally via :meth:`ReactionWheel.apply_torque`. Note the
        voltage→torque map (and with it the back-EMF damping part of ``D``)
        lives in the motor — see :meth:`state_derivative` for the composed
        voltage-input dynamics.
        """
        x = np.asarray(state, dtype=np.float64)
        if x.shape != (N_X,):
            raise ValueError(f"state must have shape ({N_X},), got {x.shape}")
        theta_p, theta_p_dot, _, theta_w_dot = x
        torque_net = self.wheel.apply_torque(float(input_torque), theta_w_dot)
        gravity_torque = (
            self.pendulum_mass * self.gravity * self.pendulum_length * math.sin(theta_p)
        )
        theta_p_ddot = (
            gravity_torque - self.pivot_friction * theta_p_dot - torque_net
        ) / self.body_inertia
        theta_w_ddot = torque_net / self.wheel.inertia - theta_p_ddot
        return np.array(
            [theta_p_dot, theta_p_ddot, theta_w_dot, theta_w_ddot], dtype=np.float64
        )

    def state_derivative(self, state, voltage: float) -> np.ndarray:
        """Nonlinear ``ẋ = f(x, u)`` with the **voltage** as input ``u``.

        Composes the motor's (saturated, back-EMF-aware) voltage→torque map
        with :meth:`nonlinear_dynamics` — the closed form whose linearisation
        at the upright equilibrium is :meth:`linearize`'s ``(A, B)``.
        """
        x = np.asarray(state, dtype=np.float64)
        if x.shape != (N_X,):
            raise ValueError(f"state must have shape ({N_X},), got {x.shape}")
        torque = self.motor.compute_torque(float(voltage), x[3])
        return self.nonlinear_dynamics(x, torque)

    # ------------------------------------------------------------------ #
    # linearisation
    # ------------------------------------------------------------------ #
    def linearize(self, operating_point=None) -> StateSpaceModel:
        """Continuous-time ``(A, B, C, D)`` at the upright equilibrium.

        Assembles ``model.md``'s explicit matrices ("The matrices to build")
        verbatim, with the lumped scalars ``E`` and ``D`` from the components.
        Only the upright equilibrium ``θ = 0`` (rates zero, wheel angle free)
        is supported — the model is a small-angle approximation. ``C`` selects
        the sensed outputs ``[theta_p, theta_w_dot]`` (``model.md``
        "Measurement model"; ``n_y = 2``) with zero feedthrough.

        Raises
        ------
        ValueError
            If ``operating_point`` is given and is not the upright equilibrium.
        """
        if operating_point is not None:
            x0 = np.asarray(operating_point, dtype=np.float64)
            if x0.shape != (N_X,):
                raise ValueError(
                    f"operating_point must have shape ({N_X},), got {x0.shape}"
                )
            # wheel angle (index 2) is free: column 3 of A is zero (model.md)
            if max(abs(x0[0]), abs(x0[1]), abs(x0[3])) > ATOL:
                raise ValueError(
                    "only the upright equilibrium theta_p = rates = 0 is "
                    "supported (model.md is a small-angle approximation)"
                )
        I_b = self.body_inertia
        I_w = self.wheel.inertia
        E = self.voltage_to_torque_gain
        D_lump = self.lumped_damping
        mgl = self.pendulum_mass * self.gravity * self.pendulum_length
        coupling = (I_w + I_b) / (I_w * I_b)
        A = np.array(
            [
                [0.0, 1.0, 0.0, 0.0],
                [mgl / I_b, -self.pivot_friction / I_b, 0.0, D_lump / I_b],
                [0.0, 0.0, 0.0, 1.0],
                [-mgl / I_b, self.pivot_friction / I_b, 0.0, -D_lump * coupling],
            ],
            dtype=np.float64,
        )
        B = np.array(
            [[0.0], [-E / I_b], [0.0], [E * coupling]], dtype=np.float64
        )
        C = np.array(
            [[1.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 1.0]], dtype=np.float64
        )
        D = np.zeros((2, 1), dtype=np.float64)
        return StateSpaceModel(A, B, C, D)
