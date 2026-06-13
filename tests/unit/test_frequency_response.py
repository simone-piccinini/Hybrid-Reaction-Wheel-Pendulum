"""Unit tests for dynamics.frequency_response (state space → Laplace → Bode)."""

import numpy as np
import pytest

from inverted_pendulum.core.types import StateSpaceModel
from inverted_pendulum.dynamics.frequency_response import (
    BodeData,
    bode,
    log_frequencies,
    transfer_function,
)
from inverted_pendulum.numerics.linalg import SingularMatrixError
from inverted_pendulum.physical.motor import DCMotor
from inverted_pendulum.physical.pendulum import ReactionWheelPendulum
from inverted_pendulum.physical.wheel import ReactionWheel


def first_order(a=2.0, b=3.0, c=1.0, d=0.0):
    """Continuous SISO ``ẋ = −a x + b u, y = c x + d u``  ->  G(s) = c·b/(s+a) + d."""
    return StateSpaceModel([[-a]], [[b]], [[c]], [[d]])


def pendulum_model():
    plant = ReactionWheelPendulum(
        pendulum_mass=0.3, pendulum_length=0.15, body_inertia=0.02,
        pivot_friction=0.01,
        wheel=ReactionWheel(mass=0.1, radius=0.05, inertia=1e-4,
                            friction_coefficient=1e-4),
        motor=DCMotor(resistance=2.0, torque_constant=0.05,
                      back_emf_constant=0.05),
    )
    return plant.linearize()


# --------------------------------------------------------------------------- #
# the transfer function — analytic first-order check
# --------------------------------------------------------------------------- #
def test_first_order_matches_closed_form():
    # G(jω) = b/(jω + a): magnitude b/sqrt(a²+ω²), phase −atan2(ω, a)
    a, b = 2.0, 3.0
    model = first_order(a=a, b=b)
    omega = np.array([0.0, 0.5, 1.0, 2.0, 10.0])
    G = transfer_function(model, omega)[:, 0, 0]
    expected = b / (1j * omega + a)
    np.testing.assert_allclose(G, expected, rtol=1e-12, atol=1e-12)


def test_first_order_dc_gain():
    # at ω = 0, G(0) = b/a (real)
    model = first_order(a=4.0, b=8.0)
    G0 = transfer_function(model, 0.0)[0, 0, 0]
    assert G0 == pytest.approx(2.0)
    assert abs(G0.imag) < 1e-12


def test_feedthrough_adds_D():
    # G(s) = b/(s+a) + d  ->  at high ω, G → d
    model = first_order(a=1.0, b=1.0, d=0.5)
    G = transfer_function(model, 1e6)[0, 0, 0]
    assert G.real == pytest.approx(0.5, abs=1e-3)


def test_shape_and_dtype():
    model = pendulum_model()
    G = transfer_function(model, log_frequencies(1e-2, 1e2, 20))
    assert G.shape == (20, model.n_y, model.n_u)
    assert G.dtype == np.complex128


def test_scalar_omega_returns_length_one():
    G = transfer_function(first_order(), 3.0)
    assert G.shape == (1, 1, 1)


# --------------------------------------------------------------------------- #
# poles on the contour — the integrator
# --------------------------------------------------------------------------- #
def test_pure_integrator_is_singular_at_dc():
    # ẋ = u, y = x  ->  G(s) = 1/s, a pole at s = 0: (0·I − A) is singular
    integrator = StateSpaceModel([[0.0]], [[1.0]], [[1.0]], [[0.0]])
    with pytest.raises(SingularMatrixError):
        transfer_function(integrator, 0.0)
    # away from the pole it is fine: |1/(jω)| = 1/ω
    G = transfer_function(integrator, 2.0)[0, 0, 0]
    assert abs(G) == pytest.approx(0.5)


def test_pendulum_has_an_integrator_pole_at_dc():
    # the wheel angle is an integrator (zero eigenvalue of A) -> singular at ω=0
    with pytest.raises(SingularMatrixError):
        transfer_function(pendulum_model(), 0.0)


# --------------------------------------------------------------------------- #
# discrete models — z = e^{jωΔt}
# --------------------------------------------------------------------------- #
def test_discrete_uses_unit_circle_point():
    cont = first_order(a=2.0, b=3.0)
    dt = 0.05
    disc = cont.discretize(dt)
    omega = np.array([0.1, 1.0, 5.0])
    G = transfer_function(disc, omega)[:, 0, 0]
    z = np.exp(1j * omega * dt)
    expected = disc.C[0, 0] * disc.B[0, 0] / (z - disc.A[0, 0]) + disc.D[0, 0]
    np.testing.assert_allclose(G, expected, rtol=1e-10, atol=1e-12)


# --------------------------------------------------------------------------- #
# bode reduction
# --------------------------------------------------------------------------- #
def test_bode_is_magnitude_and_phase_of_the_response():
    model = pendulum_model()
    omega = log_frequencies(1e-2, 1e2, 50)
    data = bode(model, omega, output_index=0, input_index=0)
    assert isinstance(data, BodeData)
    channel = transfer_function(model, omega)[:, 0, 0]
    np.testing.assert_allclose(data.magnitude_db, 20 * np.log10(np.abs(channel)))
    # phase matches angle up to the unwrap (compare wrapped to wrapped)
    np.testing.assert_allclose(
        np.angle(np.exp(1j * np.radians(data.phase_deg))),
        np.angle(channel), atol=1e-9,
    )


def test_bode_first_order_magnitude_slope():
    # a first-order low-pass rolls off at −20 dB/decade well above the corner
    model = first_order(a=1.0, b=1.0)
    data = bode(model, np.array([100.0, 1000.0]))  # a decade, both ≫ corner
    slope = data.magnitude_db[1] - data.magnitude_db[0]
    assert slope == pytest.approx(-20.0, abs=0.5)


def test_bode_channel_selection_validates():
    model = pendulum_model()
    omega = log_frequencies(1e-1, 1e1, 5)
    bode(model, omega, output_index=1, input_index=0)  # valid: n_y=2
    with pytest.raises(ValueError):
        bode(model, omega, output_index=2)  # only 2 outputs
    with pytest.raises(ValueError):
        bode(model, omega, input_index=1)   # only 1 input


# --------------------------------------------------------------------------- #
# guards and the frequency grid
# --------------------------------------------------------------------------- #
def test_transfer_function_guards():
    with pytest.raises(TypeError):
        transfer_function("not a model", 1.0)
    with pytest.raises(ValueError):
        transfer_function(first_order(), -1.0)  # negative ω


def test_log_frequencies():
    grid = log_frequencies(1e-2, 1e2, 5)
    assert grid.shape == (5,)
    assert grid[0] == pytest.approx(1e-2)
    assert grid[-1] == pytest.approx(1e2)
    assert np.all(np.diff(grid) > 0)
    # log-spaced: equal ratios between consecutive points
    ratios = grid[1:] / grid[:-1]
    np.testing.assert_allclose(ratios, ratios[0])


def test_log_frequencies_guards():
    with pytest.raises(ValueError):
        log_frequencies(0.0, 1.0)
    with pytest.raises(ValueError):
        log_frequencies(10.0, 1.0)
    with pytest.raises(ValueError):
        log_frequencies(1e-2, 1e2, 1)
