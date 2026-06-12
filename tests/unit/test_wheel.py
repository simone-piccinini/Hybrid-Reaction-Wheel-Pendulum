"""Unit tests for physical.wheel.ReactionWheel (flywheel with bearing friction)."""

import dataclasses

import pytest

from inverted_pendulum.physical.wheel import ReactionWheel


def make(**overrides):
    params = dict(mass=0.1, radius=0.05, inertia=1e-4)
    params.update(overrides)
    return ReactionWheel(**params)


# --------------------------------------------------------------------------- #
# net torque
# --------------------------------------------------------------------------- #
def test_apply_torque_frictionless_is_identity():
    w = make(friction_coefficient=0.0)
    assert w.apply_torque(torque=0.3, angular_velocity=50.0) == pytest.approx(0.3)


def test_apply_torque_subtracts_viscous_friction():
    # τ_w = τ − b_w·ω
    w = make(friction_coefficient=2e-4)
    assert w.apply_torque(0.3, 100.0) == pytest.approx(0.3 - 2e-4 * 100.0)


def test_friction_opposes_motion_in_both_directions():
    w = make(friction_coefficient=1e-3)
    # spinning forward: friction reduces net torque; backward: increases it
    assert w.apply_torque(0.0, 10.0) == pytest.approx(-1e-2)
    assert w.apply_torque(0.0, -10.0) == pytest.approx(+1e-2)


def test_apply_torque_at_rest_passes_torque_through():
    w = make(friction_coefficient=5e-3)
    assert w.apply_torque(0.7, 0.0) == pytest.approx(0.7)


# --------------------------------------------------------------------------- #
# angular momentum
# --------------------------------------------------------------------------- #
def test_angular_momentum():
    w = make(inertia=1e-4)
    assert w.angular_momentum(200.0) == pytest.approx(1e-4 * 200.0)


def test_angular_momentum_sign_follows_velocity():
    w = make()
    assert w.angular_momentum(-30.0) == pytest.approx(-w.angular_momentum(30.0))


# --------------------------------------------------------------------------- #
# validation / immutability
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    "bad",
    [
        dict(mass=0.0),
        dict(mass=-0.1),
        dict(radius=0.0),
        dict(radius=-0.05),
        dict(inertia=0.0),
        dict(inertia=-1e-4),
        dict(friction_coefficient=-1e-4),
        # rigid-body bound: I_w ≤ m·r² (here m·r² = 0.1·0.0025 = 2.5e-4)
        dict(inertia=3e-4),
    ],
)
def test_validation_rejects_bad_params(bad):
    with pytest.raises(ValueError):
        make(**bad)


def test_inertia_at_rigid_body_bound_is_accepted():
    # all mass at the rim (thin hoop): I_w = m·r² exactly
    w = make(inertia=0.1 * 0.05**2)
    assert w.inertia == pytest.approx(2.5e-4)


def test_default_friction_is_zero():
    assert make().friction_coefficient == 0.0


def test_frozen():
    w = make()
    with pytest.raises(dataclasses.FrozenInstanceError):
        w.inertia = 1.0
