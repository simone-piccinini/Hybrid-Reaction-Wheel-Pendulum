"""Reaction wheel — the inertia the motor spins to generate reaction torque.

The wheel stores angular momentum ``L = I_w · ω``; accelerating it produces the
equal-and-opposite reaction torque on the pendulum body that is the system's
only control authority (``docs/theory/model.md``, "Where it comes from").
Viscous bearing friction ``b_w · ω`` opposes the wheel's rotation relative to
the pendulum; ``model.md`` lumps it into ``D = K_t K_e / R_a + b_w`` together
with the motor's back-EMF damping.

Symbols ↔ code names follow ``docs/theory/notation.md`` §2 (``I_w`` ↔
``inertia`` on this class, exposed as ``wheel_inertia`` by the plant). This
module depends on the standard library only.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ReactionWheel:
    """A rigid flywheel with viscous bearing friction (``class_diagram.md``).

    Parameters
    ----------
    mass : float
        Wheel mass (kg), > 0.
    radius : float
        Outer radius (m), > 0.
    inertia : float
        Moment of inertia ``I_w`` about the spin axis (kg·m²), > 0 (it is a
        denominator in ``model.md``'s matrices). Must satisfy the rigid-body
        bound ``I_w ≤ mass · radius²`` (all mass at the rim).
    friction_coefficient : float
        Viscous bearing-friction coefficient ``b_w`` (N·m·s/rad), ≥ 0
        (``model.md``'s wheel friction, lumped into ``D``).
    """

    mass: float
    radius: float
    inertia: float
    friction_coefficient: float = 0.0

    def __post_init__(self) -> None:
        for name in ("mass", "radius", "inertia", "friction_coefficient"):
            object.__setattr__(self, name, float(getattr(self, name)))
        if self.mass <= 0.0:
            raise ValueError("mass must be > 0")
        if self.radius <= 0.0:
            raise ValueError("radius must be > 0")
        if self.inertia <= 0.0:
            raise ValueError("inertia I_w must be > 0 (it is a denominator)")
        if self.inertia > self.mass * self.radius**2:
            raise ValueError(
                "inertia I_w exceeds the rigid-body bound mass·radius² "
                "(all mass at the rim)"
            )
        if self.friction_coefficient < 0.0:
            raise ValueError("friction_coefficient b_w must be >= 0")

    def apply_torque(self, torque: float, angular_velocity: float) -> float:
        """Net torque on the wheel: ``τ_w = τ − b_w · ω`` (``model.md``).

        ``torque`` is the electromagnetic motor torque on the wheel;
        ``angular_velocity`` is the wheel rate ``ω`` relative to the pendulum
        (the rate the bearing friction acts on). Both the motor torque and the
        friction are internal pendulum–wheel torques, so ``−τ_w`` is the
        reaction on the pendulum body.
        """
        return float(torque) - self.friction_coefficient * float(angular_velocity)

    def angular_momentum(self, angular_velocity: float) -> float:
        """Angular momentum ``L = I_w · ω`` stored in the wheel."""
        return self.inertia * float(angular_velocity)
