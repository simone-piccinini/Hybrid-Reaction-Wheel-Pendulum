"""DC motor actuator model.

The brushless drive is modelled as a brushed DC motor in the **steady-state**
(inductance-neglected) regime, consistent with the 4-state plant of
``docs/theory/model.md`` (the state ``[θ, θ̇, φ, φ̇]`` carries no armature-current
state). With ``L_m → 0`` the armature current is algebraic,

    i = (V_m − K_e · ω) / R_a,                       (Ohm's law, back-EMF K_e·ω)

and the electromagnetic torque on the wheel is

    τ = K_t · i = (K_t / R_a) V_m − (K_t K_e / R_a) ω.

The first coefficient is ``model.md``'s ``E = K_t / R_a`` (voltage → torque); the
second, ``K_t K_e / R_a``, is the back-EMF damping lumped into ``model.md``'s
``D = K_t K_e / R_a + b_w`` together with wheel friction.

Symbols ↔ code names follow ``docs/theory/notation.md`` §2. This module depends
on ``numpy`` and the standard library only.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class DCMotor:
    """A steady-state DC motor (``notation.md`` §2; ``class_diagram.md``).

    Parameters
    ----------
    resistance : float
        Armature resistance ``R_a`` (Ω), must be > 0 (it is a denominator).
    torque_constant : float
        Torque constant ``K_t`` (N·m/A), > 0.
    back_emf_constant : float
        Back-EMF constant ``K_e`` (V·s/rad), > 0.
    inductance : float
        Armature inductance ``L_m`` (H). Stored for completeness; the
        steady-state torque model neglects it (``model.md``).
    rotor_inertia : float
        Rotor inertia (kg·m²); the wheel inertia carries the mechanical dynamics.
    friction_coefficient : float
        Motor viscous-friction coefficient (N·m·s/rad).
    max_voltage : float
        Voltage saturation limit ``V_{m,max}`` (V), > 0 (default ``inf``).
    max_current : float
        Current saturation limit (A), > 0 (default ``inf``).
    """

    resistance: float
    torque_constant: float
    back_emf_constant: float
    inductance: float = 0.0
    rotor_inertia: float = 0.0
    friction_coefficient: float = 0.0
    max_voltage: float = math.inf
    max_current: float = math.inf

    def __post_init__(self) -> None:
        for name in (
            "resistance",
            "torque_constant",
            "back_emf_constant",
            "inductance",
            "rotor_inertia",
            "friction_coefficient",
            "max_voltage",
            "max_current",
        ):
            object.__setattr__(self, name, float(getattr(self, name)))
        if self.resistance <= 0.0:
            raise ValueError("resistance R_a must be > 0 (it is a denominator)")
        if self.torque_constant <= 0.0:
            raise ValueError("torque_constant K_t must be > 0")
        if self.back_emf_constant <= 0.0:
            raise ValueError("back_emf_constant K_e must be > 0")
        if self.max_voltage <= 0.0:
            raise ValueError("max_voltage must be > 0")
        if self.max_current <= 0.0:
            raise ValueError("max_current must be > 0")
        if min(self.inductance, self.rotor_inertia, self.friction_coefficient) < 0.0:
            raise ValueError("inductance, rotor_inertia, friction must be >= 0")

    def saturate_voltage(self, voltage: float) -> float:
        """Clamp the applied voltage to ``[−V_{m,max}, +V_{m,max}]``."""
        return float(np.clip(voltage, -self.max_voltage, self.max_voltage))

    def back_emf(self, angular_velocity: float) -> float:
        """Back-EMF voltage ``K_e · ω`` induced by the wheel's angular velocity."""
        return self.back_emf_constant * float(angular_velocity)

    def compute_current(self, voltage: float, back_emf: float) -> float:
        """Steady-state armature current ``i = (V_m − e_b) / R_a``, current-limited.

        Inductance is neglected (``model.md``); the result is clamped to
        ``[−max_current, +max_current]``. ``back_emf`` is the back-EMF voltage
        (see :meth:`back_emf`).
        """
        current = (float(voltage) - float(back_emf)) / self.resistance
        return float(np.clip(current, -self.max_current, self.max_current))

    def compute_torque(self, voltage: float, angular_velocity: float) -> float:
        """Electromagnetic torque ``τ = K_t · i`` on the wheel.

        Applies voltage saturation, forms the back-EMF ``K_e·ω``, takes the
        (current-limited) armature current, and scales by ``K_t``. In the
        unsaturated regime this equals ``E·V_m − (K_t K_e / R_a)·ω`` with
        ``E = K_t / R_a`` (``model.md``).
        """
        saturated = self.saturate_voltage(voltage)
        current = self.compute_current(saturated, self.back_emf(angular_velocity))
        return self.torque_constant * current

    @property
    def voltage_to_torque_gain(self) -> float:
        """``E = K_t / R_a`` — the no-load voltage→torque gain (``model.md``)."""
        return self.torque_constant / self.resistance

    @property
    def back_emf_damping(self) -> float:
        """``K_t K_e / R_a`` — the back-EMF damping coefficient (``model.md``'s D term)."""
        return self.torque_constant * self.back_emf_constant / self.resistance
