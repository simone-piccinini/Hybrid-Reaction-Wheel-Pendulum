"""Run persistence — the reproducibility record of AGENTS §7.

"ExperimentManager records: git commit hash, full config, seed, metrics, and
trajectories for every run." This module is the on-disk half of that contract.
``io`` may depend on ``core`` only (``dependency_rules.md`` §2); a
:class:`SimulationResult` is a ``core`` type, so it can be persisted here.

Named ``run_logging`` rather than ``logging`` to avoid shadowing the standard
library ``logging`` module on import.

A run directory ends up holding:

    <run_dir>/
      metadata.json      git hash, seed, timestamp, config, metrics, summary
      trajectories.npz   time / true_states / estimated_states / controls / measurements
      history.npz        X (evaluated θ) and y (their costs) — the BO dataset

Everything written is plain JSON or ``.npz`` so a run can be reloaded with the
standard library and NumPy alone.
"""

from __future__ import annotations

import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from ..core.types import SimulationResult


def current_git_hash(default: str = "unknown") -> str:
    """The current commit hash, or ``default`` outside a git work tree.

    Reproducibility (AGENTS §7): every run records the exact code revision.
    Never raises — a missing repo or git binary degrades to ``default``.
    """
    try:
        completed = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True, text=True, check=True,
        )
        return completed.stdout.strip() or default
    except (subprocess.CalledProcessError, FileNotFoundError, OSError):
        return default


def save_run(
    directory,
    *,
    config: dict,
    seed: int,
    metrics: dict,
    result: SimulationResult | None = None,
    history: tuple[np.ndarray, np.ndarray] | None = None,
    summary: dict | None = None,
    git_hash: str | None = None,
) -> Path:
    """Write the full reproducibility record for one experiment run.

    Parameters
    ----------
    directory : path-like
        Destination run directory (created, parents included).
    config : dict
        The raw configuration that produced the run.
    seed : int
        The master seed (AGENTS §7).
    metrics : dict
        Scalar metrics of the representative run (overshoot, settling time, …).
    result : SimulationResult, optional
        The representative rollout whose trajectories are saved.
    history : (X, y), optional
        The Bayesian-optimisation dataset (evaluated points and their costs).
    summary : dict, optional
        Extra free-form summary (e.g. best config, best observed cost).
    git_hash : str, optional
        Code revision; resolved with :func:`current_git_hash` if omitted.

    Returns
    -------
    pathlib.Path
        The run directory written to.
    """
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)

    metadata = {
        "git_hash": current_git_hash() if git_hash is None else git_hash,
        "seed": int(seed),
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "config": config,
        "metrics": metrics,
        "summary": summary or {},
    }
    with (directory / "metadata.json").open("w") as handle:
        json.dump(metadata, handle, indent=2, default=_json_default)

    if result is not None:
        np.savez(
            directory / "trajectories.npz",
            time=result.time,
            true_states=result.true_states,
            estimated_states=result.estimated_states,
            controls=result.controls,
            measurements=result.measurements,
            seed=result.seed,
            diverged=result.diverged,
        )
    if history is not None:
        X, y = history
        np.savez(directory / "history.npz", X=np.asarray(X), y=np.asarray(y))

    return directory


def load_metadata(directory) -> dict:
    """Read back the ``metadata.json`` of a saved run (for inspection/tests)."""
    with (Path(directory) / "metadata.json").open("r") as handle:
        return json.load(handle)


def _json_default(value):
    """Make NumPy scalars/arrays JSON-serialisable."""
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(f"not JSON-serialisable: {type(value)!r}")
