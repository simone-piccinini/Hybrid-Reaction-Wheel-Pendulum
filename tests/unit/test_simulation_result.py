"""Unit tests for core.types.SimulationResult."""

import numpy as np
import pytest

from inverted_pendulum.core.types import LQGConfig, SimulationResult

N_X, N_U, N_Y = 4, 1, 2


def make_config():
    return LQGConfig(
        np.diag([2.0, 3.0, 4.0, 5.0]),
        np.diag([1.5]),
        np.diag([0.1, 0.2, 0.3, 0.4]),
        np.diag([0.05, 0.06]),
    )


def make_result(T=10, diverged=False, seed=42, **overrides):
    rng = np.random.default_rng(0)
    fields = dict(
        time=np.linspace(0.0, 1.0, T),
        true_states=rng.standard_normal((T, N_X)),
        estimated_states=rng.standard_normal((T, N_X)),
        controls=rng.standard_normal((T, N_U)),
        measurements=rng.standard_normal((T, N_Y)),
        seed=seed,
        diverged=diverged,
        config=make_config(),
    )
    fields.update(overrides)
    return SimulationResult(**fields)


def test_valid_result_fields_and_properties():
    r = make_result(T=10)
    assert r.horizon == 10
    assert (r.n_x, r.n_u, r.n_y) == (4, 1, 2)
    assert r.seed == 42 and r.diverged is False
    for arr in (r.time, r.true_states, r.controls, r.measurements):
        assert arr.dtype == np.float64


def test_diverged_flag_allowed():
    assert make_result(diverged=True).diverged is True


def test_arrays_read_only():
    r = make_result()
    with pytest.raises(ValueError):
        r.true_states[0, 0] = 9.0
    with pytest.raises(ValueError):
        r.time[0] = 9.0


@pytest.mark.parametrize(
    "field,bad",
    [
        ("true_states", np.zeros((9, N_X))),     # T mismatch
        ("estimated_states", np.zeros((10, 3))),  # n_x mismatch
        ("controls", np.zeros((10, 2))),          # n_u mismatch
        ("measurements", np.zeros((10, 1))),      # n_y mismatch
    ],
)
def test_shape_validation(field, bad):
    with pytest.raises(ValueError):
        make_result(T=10, **{field: bad})


def test_time_must_be_1d_nonempty():
    with pytest.raises(ValueError):
        make_result(time=np.zeros((10, 1)))  # 2-D time


def test_seed_must_be_integer():
    with pytest.raises(TypeError):
        make_result(seed=3.5)
    with pytest.raises(TypeError):
        make_result(seed=True)  # bool is not a valid seed


def test_diverged_must_be_bool():
    with pytest.raises(TypeError):
        make_result(diverged=1)


def test_config_must_be_lqgconfig():
    with pytest.raises(TypeError):
        make_result(config={"not": "a config"})
