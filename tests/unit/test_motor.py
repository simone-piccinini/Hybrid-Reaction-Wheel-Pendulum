"""Unit tests for physical.motor.DCMotor (steady-state DC motor model)."""

import dataclasses
import math

import numpy as np
import pytest

from inverted_pendulum.physical.motor import DCMotor


def make(**overrides):
    params = dict(resistance=2.0, torque_constant=0.5, back_emf_constant=0.1)
    params.update(overrides)
    return DCMotor(**params)


# --------------------------------------------------------------------------- #
# Ohm's law / current
# --------------------------------------------------------------------------- #
def test_compute_current_ohms_law():
    m = make(resistance=2.0)
    assert m.compute_current(voltage=10.0, back_emf=4.0) == pytest.approx((10.0 - 4.0) / 2.0)


def test_current_saturation():
    m = make(resistance=1.0, max_current=5.0)
    assert m.compute_current(voltage=100.0, back_emf=0.0) == pytest.approx(5.0)
    assert m.compute_current(voltage=-100.0, back_emf=0.0) == pytest.approx(-5.0)


def test_back_emf():
    m = make(back_emf_constant=0.1)
    assert m.back_emf(10.0) == pytest.approx(1.0)


# --------------------------------------------------------------------------- #
# torque
# --------------------------------------------------------------------------- #
def test_compute_torque_known_value():
    # K_t=0.5, R=2, K_e=0.1, V=10, ω=10 -> e_b=1, i=(10-1)/2=4.5, τ=0.5*4.5=2.25
    m = make()
    assert m.compute_torque(voltage=10.0, angular_velocity=10.0) == pytest.approx(2.25)


def test_torque_at_zero_velocity_is_E_times_voltage():
    # ω=0 -> τ = (K_t/R_a) V = E·V
    m = make()
    V = 7.0
    assert m.compute_torque(V, 0.0) == pytest.approx(m.voltage_to_torque_gain * V)


def test_torque_matches_model_md_lumped_form_unsaturated():
    # Unsaturated: τ = E·V − (K_t K_e / R_a)·ω
    m = make()
    V, w = 6.0, 12.0
    expected = m.voltage_to_torque_gain * V - m.back_emf_damping * w
    assert m.compute_torque(V, w) == pytest.approx(expected)


def test_voltage_saturation_in_torque():
    m = make(max_voltage=5.0)
    # voltage clamped to 5 before the torque calculation
    assert m.compute_torque(100.0, 0.0) == pytest.approx(m.compute_torque(5.0, 0.0))


def test_lumped_scalar_properties():
    m = make(torque_constant=0.5, back_emf_constant=0.1, resistance=2.0)
    assert m.voltage_to_torque_gain == pytest.approx(0.25)        # E = K_t/R_a
    assert m.back_emf_damping == pytest.approx(0.5 * 0.1 / 2.0)   # K_t K_e / R_a


# --------------------------------------------------------------------------- #
# validation / immutability
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    "bad",
    [
        dict(resistance=0.0),
        dict(resistance=-1.0),
        dict(torque_constant=0.0),
        dict(back_emf_constant=-0.1),
        dict(max_voltage=0.0),
        dict(max_current=-1.0),
        dict(inductance=-1.0),
    ],
)
def test_validation_rejects_bad_params(bad):
    with pytest.raises(ValueError):
        make(**bad)


def test_defaults_are_unbounded_saturation():
    m = make()
    assert m.max_voltage == math.inf and m.max_current == math.inf
    # with infinite limits a huge voltage is not clamped
    assert m.compute_torque(1e6, 0.0) == pytest.approx(m.voltage_to_torque_gain * 1e6)


def test_frozen():
    m = make()
    with pytest.raises(dataclasses.FrozenInstanceError):
        m.resistance = 5.0
