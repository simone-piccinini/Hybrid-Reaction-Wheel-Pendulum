"""Energy-shaping swing-up controller for the reaction-wheel pendulum.

**Not part of the LQG/BO stack.** This is the *global* nonlinear maneuver that
brings the pendulum from hanging (``theta_p = pi``) up into the small-angle basin
where the LQR (``docs/theory/lqr.md``) can catch and balance it. The two run in
sequence under a switching supervisor (``scripts/swing_up.py``): swing-up until
near upright, then hand off to the balancing LQG. The linear model everything
else uses is a small-angle approximation valid only near ``theta_p = 0``
(``model.md``); swing-up lives on the *full nonlinear* plant.

Method — **energy shaping** (Astrom & Furuta, "Swinging up a pendulum by energy
control", *Automatica* 2000; adapted to the reaction wheel). The pendulum's
mechanical energy (body only; the wheel torque is the control) is

    E(theta, theta_dot) = 1/2 I_b theta_dot^2 + m g l cos(theta),

with ``theta`` measured from upright, so the upright equilibrium sits at the
maximum ``E_up = m g l`` (``theta = 0``, ``theta_dot = 0``) and hanging at
``-m g l``. Writing the energy error ``E_tilde = E - E_up``, the body equation of
motion (``pendulum.py``: ``I_b theta_ddot = m g l sin theta - b_p theta_dot -
tau_w``) gives

    dE_tilde/dt = -b_p theta_dot^2 - tau_w theta_dot,

so choosing the net wheel torque ``tau_w = k E_tilde theta_dot`` makes
``dE_tilde/dt = -b_p theta_dot^2 - k E_tilde theta_dot^2 <= 0`` in magnitude of
``E_tilde`` — it drives the energy monotonically toward the upright level for any
gain ``k > 0``. Inverting the (unsaturated) voltage->torque map ``tau_w ~ E V``
with ``E = K_t / R_a`` (``motor.py``) yields the command

    V = k E_tilde theta_dot / E,     saturated to the motor rail.

Because this build's peak actuator torque is far below the gravity torque at the
horizontal (``model.md``; ~0.044 vs ~0.184 N*m on the measured plant), a single
lift is impossible: the saturated law becomes effectively bang-bang and pumps
energy over several swings (resonant swing-up). The wheel necessarily spins up as
it pumps — that momentum is what the balancer inherits at the catch, and why the
hand-off must happen while the wheel is not yet saturated.

Kept to plain scalars (numpy/math only) so ``control`` stays within its import
layer (``dependency_rules.md`` §2); the script extracts the plant parameters and
constructs it.
"""

from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class EnergySwingUpController:
    """Energy-shaping swing-up law ``V = k (E - E_up) theta_dot / E`` (saturated).

    Parameters (all SI; ``notation.md`` §2)
    ----------
    mass : float
        Pendulum mass ``m`` (kg), > 0.
    length : float
        Pivot->centre-of-mass distance ``l`` (m), > 0.
    gravity : float
        Gravitational acceleration ``g`` (m/s^2), > 0.
    body_inertia : float
        Pendulum inertia about the pivot ``I_b`` (kg*m^2), > 0.
    voltage_to_torque_gain : float
        ``E = K_t / R_a`` (N*m/V), > 0 — the motor's no-load voltage->torque gain.
    max_voltage : float
        Rail ``V_max`` (V), > 0; the command is clamped to ``[-V_max, +V_max]``.
    energy_gain : float
        Pumping gain ``k`` (V per J*rad/s, up to the ``E`` factor), > 0. Larger
        ``k`` pumps harder; with a weak actuator the command saturates and the law
        is bang-bang, so ``k`` mostly sets how early saturation kicks in.
    """

    mass: float
    length: float
    gravity: float
    body_inertia: float
    voltage_to_torque_gain: float
    max_voltage: float
    energy_gain: float

    def __post_init__(self) -> None:
        for name in ("mass", "length", "gravity", "body_inertia",
                     "voltage_to_torque_gain", "max_voltage", "energy_gain"):
            object.__setattr__(self, name, float(getattr(self, name)))
        if min(self.mass, self.length, self.gravity, self.body_inertia,
               self.voltage_to_torque_gain, self.max_voltage, self.energy_gain) <= 0.0:
            raise ValueError("all swing-up parameters must be > 0")

    @property
    def upright_energy(self) -> float:
        """The energy at the upright equilibrium, ``E_up = m g l`` (J)."""
        return self.mass * self.gravity * self.length

    def energy(self, theta_p: float, theta_p_dot: float) -> float:
        """Pendulum mechanical energy ``E = 1/2 I_b theta_dot^2 + m g l cos(theta)`` (J).

        ``theta_p`` is measured from upright (``0`` = up, ``pi`` = hanging); the
        cosine makes ``E`` periodic, so an unwrapped angle is fine.
        """
        return (0.5 * self.body_inertia * float(theta_p_dot) ** 2
                + self.upright_energy * math.cos(float(theta_p)))

    def energy_error(self, theta_p: float, theta_p_dot: float) -> float:
        """Energy relative to upright, ``E_tilde = E - E_up`` (<= 0 below the top)."""
        return self.energy(theta_p, theta_p_dot) - self.upright_energy

    def compute_voltage(self, theta_p: float, theta_p_dot: float) -> float:
        """The swing-up command ``V = k E_tilde theta_dot / E``, saturated to the rail.

        The sign always pumps energy toward the upright level: the reaction
        torque on the body, ``-E*V = -k E_tilde theta_dot``, has the sign of
        ``theta_dot`` whenever ``E_tilde < 0`` (below the top), so it accelerates
        the pendulum in its direction of motion and adds energy each swing.
        Returns ``0`` at the upright energy level (``E_tilde = 0``) or at rest
        (``theta_dot = 0``), the natural fixed points of the law.
        """
        theta_p_dot = float(theta_p_dot)
        e_tilde = self.energy_error(theta_p, theta_p_dot)
        voltage = self.energy_gain * e_tilde * theta_p_dot / self.voltage_to_torque_gain
        if voltage > self.max_voltage:
            return self.max_voltage
        if voltage < -self.max_voltage:
            return -self.max_voltage
        return voltage
