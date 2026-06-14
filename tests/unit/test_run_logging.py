"""Unit tests for io.run_logging (git hash + reproducibility record)."""

import numpy as np

from inverted_pendulum.io.run_logging import (
    current_git_hash,
    load_metadata,
    save_run,
)


def test_current_git_hash_is_a_string():
    h = current_git_hash()
    assert isinstance(h, str) and len(h) > 0


def test_current_git_hash_default_outside_repo(tmp_path, monkeypatch):
    # force the git call to fail -> the documented default, never an exception
    import subprocess

    def boom(*a, **k):
        raise FileNotFoundError("no git")

    monkeypatch.setattr(subprocess, "run", boom)
    assert current_git_hash(default="none") == "none"


def test_save_run_writes_the_bundle(tmp_path, make_result):
    result = make_result([0.05, 0.02, 0.0], dt=0.01)
    out = save_run(
        tmp_path / "run1",
        config={"seed": 7, "plant": {"mass": 0.3}},
        seed=7,
        metrics={"objective": 1.5, "diverged": False},
        result=result,
        history=(np.zeros((3, 11)), np.array([1.0, 2.0, 3.0])),
        summary={"best_theta": np.arange(11.0)},
        git_hash="deadbeef",
    )
    assert (out / "metadata.json").is_file()
    assert (out / "trajectories.npz").is_file()
    assert (out / "history.npz").is_file()


def test_metadata_round_trips_with_numpy(tmp_path, make_result):
    save_run(
        tmp_path / "run2",
        config={"a": np.array([1.0, 2.0])},  # numpy in the config dict
        seed=np.int64(11),
        metrics={"score": np.float64(0.25)},
        summary={"vec": np.arange(3)},
        git_hash="abc123",
    )
    md = load_metadata(tmp_path / "run2")
    assert md["git_hash"] == "abc123"
    assert md["seed"] == 11
    assert md["metrics"]["score"] == 0.25
    assert md["config"]["a"] == [1.0, 2.0]  # numpy serialised to a list
    assert md["summary"]["vec"] == [0, 1, 2]
    assert "timestamp_utc" in md


def test_trajectories_reload(tmp_path, make_result):
    result = make_result([0.05, 0.0, -0.01, 0.0], dt=0.02)
    save_run(tmp_path / "run3", config={}, seed=0, metrics={}, result=result)
    loaded = np.load(tmp_path / "run3" / "trajectories.npz")
    np.testing.assert_allclose(loaded["true_states"], result.true_states)
    assert int(loaded["seed"]) == result.seed
    assert bool(loaded["diverged"]) == result.diverged


def test_save_run_without_optional_artifacts(tmp_path):
    out = save_run(tmp_path / "run4", config={}, seed=1, metrics={"x": 1.0})
    assert (out / "metadata.json").is_file()
    assert not (out / "trajectories.npz").exists()
    assert not (out / "history.npz").exists()
