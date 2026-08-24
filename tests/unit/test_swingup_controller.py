"""Unit tests for control.swingup_controller.EnergySwingUpController.

Energy-shaping law V = k (E - E_up) theta_dot / E, saturated. The invariants:
E is the pendulum energy (max at upright), the command vanishes at the fixed
points (upright energy, or zero rate), and its sign always pumps energy toward
the top (reaction power into the body >= 0 below the top).
"""

import math

import numpy as np
import pytest

from inverted_pendulum.control.swingup_controller import EnergySwingUpController


def make(**overrides):
    # simple round numbers: E_up = m g l = 10, I_b = E = 1
    params = dict(mass=1.0, length=1.0, gravity=10.0, body_inertia=1.0,
                  voltage_to_torque_gain=1.0, max_voltage=100.0, energy_gain=1.0)
    params.update(overrides)
    return EnergySwingUpController(**params)


# --------------------------------------------------------------------------- #
# the energy function
# --------------------------------------------------------------------------- #
def test_upright_energy_is_mgl():
    assert make().upright_energy == pytest.approx(10.0)


def test_energy_is_max_at_upright_and_min_at_hanging():
    c = make()
    assert c.energy(0.0, 0.0) == pytest.approx(10.0)           # upright: +mgl
    assert c.energy(math.pi, 0.0) == pytest.approx(-10.0)      # hanging: -mgl
    assert c.energy_error(0.0, 0.0) == pytest.approx(0.0)
    assert c.energy_error(math.pi, 0.0) == pytest.approx(-20.0)


def test_kinetic_term_uses_half_I_theta_dot_squared():
    c = make(body_inertia=2.0)
    # at horizontal (cos = 0) the energy is purely kinetic
    assert c.energy(math.pi / 2, 3.0) == pytest.approx(0.5 * 2.0 * 9.0)


# --------------------------------------------------------------------------- #
# the command vanishes at the fixed points
# --------------------------------------------------------------------------- #
def test_no_command_at_the_upright_energy_level():
    # E_tilde = 0 -> V = 0 even while moving (on the homoclinic orbit)
    c = make()
    theta = 0.3
    theta_dot = math.sqrt(2.0 * (c.upright_energy - c.upright_energy * math.cos(theta))
                          / c.body_inertia)  # energy exactly E_up
    assert c.energy_error(theta, theta_dot) == pytest.approx(0.0, abs=1e-9)
    assert c.compute_voltage(theta, theta_dot) == pytest.approx(0.0, abs=1e-9)


def test_no_command_at_rest():
    # theta_dot = 0 -> V = 0 (the hanging equilibrium cannot self-start)
    assert make().compute_voltage(math.pi, 0.0) == pytest.approx(0.0)


# --------------------------------------------------------------------------- #
# the sign always pumps energy toward the top
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("theta,theta_dot", [
    (math.pi / 2, 2.0), (math.pi / 2, -2.0), (2.5, 0.7), (math.pi, 1.5),
])
def test_reaction_power_is_non_negative_below_the_top(theta, theta_dot):
    # body reaction torque is -E*V; the power it injects, -E*V*theta_dot, must be
    # >= 0 when below the upright energy (it adds energy every swing)
    c = make(max_voltage=100.0)
    v = c.compute_voltage(theta, theta_dot)
    power = -c.voltage_to_torque_gain * v * theta_dot
    assert c.energy_error(theta, theta_dot) < 0.0
    assert power >= 0.0


def test_command_opposes_rate_below_the_top():
    # E_tilde < 0 => V = k E_tilde theta_dot / E has the opposite sign to theta_dot
    c = make(max_voltage=100.0)
    assert c.compute_voltage(math.pi / 2, 2.0) < 0.0
    assert c.compute_voltage(math.pi / 2, -2.0) > 0.0


def test_unsaturated_value_matches_the_formula():
    c = make(max_voltage=100.0)
    theta, theta_dot = math.pi / 2, 0.5
    expected = c.energy_gain * c.energy_error(theta, theta_dot) * theta_dot \
        / c.voltage_to_torque_gain
    assert c.compute_voltage(theta, theta_dot) == pytest.approx(expected)
    assert abs(expected) < c.max_voltage  # genuinely unsaturated


# --------------------------------------------------------------------------- #
# saturation to the rail
# --------------------------------------------------------------------------- #
def test_command_saturates_to_the_rail():
    c = make(max_voltage=5.0, energy_gain=10.0)
    v = c.compute_voltage(math.pi / 2, 3.0)  # large |E_tilde * theta_dot|
    assert v == pytest.approx(-5.0)          # clamped to -V_max
    assert abs(c.compute_voltage(math.pi / 2, -3.0)) == pytest.approx(5.0)


# --------------------------------------------------------------------------- #
# validation
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("bad", [
    dict(mass=0.0), dict(length=-1.0), dict(gravity=0.0), dict(body_inertia=0.0),
    dict(voltage_to_torque_gain=0.0), dict(max_voltage=0.0), dict(energy_gain=-1.0),
])
def test_validation_rejects_non_positive_params(bad):
    with pytest.raises(ValueError):
        make(**bad)
