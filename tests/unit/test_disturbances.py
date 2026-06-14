"""Unit tests for simulation.disturbances.Disturbances (seeded true-world noise)."""

import dataclasses

import numpy as np
import pytest

from inverted_pendulum.simulation.disturbances import Disturbances


def make(**overrides):
    params = dict(process_std=[0.0, 1e-4, 0.0, 1e-2], measurement_std=[1e-3, 1e-2])
    params.update(overrides)
    return Disturbances(**params)


def test_dimensions():
    d = make()
    assert (d.n_x, d.n_y) == (4, 2)


def test_zero_std_draws_zeros():
    d = Disturbances(process_std=np.zeros(4), measurement_std=np.zeros(2))
    rng = np.random.default_rng(0)
    np.testing.assert_array_equal(d.process_noise(rng), np.zeros(4))
    np.testing.assert_array_equal(d.measurement_noise(rng), np.zeros(2))
    np.testing.assert_array_equal(d.initial_perturbation(rng), np.zeros(4))


def test_draws_are_deterministic_given_the_generator():
    d = make(initial_state_std=[1e-2, 0.0, 0.0, 0.0])
    a = np.random.default_rng(123)
    b = np.random.default_rng(123)
    np.testing.assert_array_equal(d.process_noise(a), d.process_noise(b))
    np.testing.assert_array_equal(d.measurement_noise(a), d.measurement_noise(b))
    np.testing.assert_array_equal(d.initial_perturbation(a), d.initial_perturbation(b))


def test_zero_channels_stay_zero():
    d = make()  # process_std has zeros at indices 0 and 2
    rng = np.random.default_rng(5)
    for _ in range(10):
        w = d.process_noise(rng)
        assert w[0] == 0.0 and w[2] == 0.0


def test_std_scaling_statistically():
    d = Disturbances(process_std=[2.0, 0.5, 1.0, 3.0], measurement_std=[1.0, 1.0])
    rng = np.random.default_rng(2024)
    draws = np.array([d.process_noise(rng) for _ in range(20000)])
    np.testing.assert_allclose(draws.std(axis=0), [2.0, 0.5, 1.0, 3.0], rtol=0.05)
    np.testing.assert_allclose(draws.mean(axis=0), np.zeros(4), atol=0.05)


def test_initial_std_defaults_to_zeros():
    d = make()
    np.testing.assert_array_equal(d.initial_state_std, np.zeros(4))


@pytest.mark.parametrize(
    "bad",
    [
        dict(process_std=[-1e-3, 0.0, 0.0, 0.0]),
        dict(measurement_std=[1e-3, -1e-2]),
        dict(initial_state_std=[1e-2, 0.0]),          # wrong length vs n_x
        dict(process_std=np.zeros((2, 2))),           # not a vector
    ],
)
def test_validation_rejects_bad_inputs(bad):
    with pytest.raises(ValueError):
        make(**bad)


def test_frozen_and_read_only():
    d = make()
    with pytest.raises(dataclasses.FrozenInstanceError):
        d.process_std = np.zeros(4)
    assert not d.process_std.flags.writeable
